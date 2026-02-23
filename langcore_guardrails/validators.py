"""Pluggable validators for the guardrail provider.

Each validator inspects a raw LLM output string and returns a
``ValidationResult`` indicating whether the output is acceptable.
When validation fails, the error message is injected into a
corrective prompt for retry.

New in v1.1.0
~~~~~~~~~~~~~
- ``OnFailAction`` enum — ``EXCEPTION``, ``REASK``, ``FILTER``,
  ``NOOP``.
- ``SchemaValidator`` — Pydantic model validation with strict
  types and constraints.
- ``ConfidenceThresholdValidator`` — rejects extractions below a
  confidence threshold.
- ``FieldCompletenessValidator`` — ensures required Pydantic
  fields are present and non-empty.
- ``ConsistencyValidator`` — cross-checks extracted values via
  user-supplied rules.
"""

from __future__ import annotations

import abc
import enum
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable
import copy

import jsonschema
from json_repair import repair_json

__all__ = [
    "ConfidenceThresholdValidator",
    "ConsistencyValidator",
    "FieldCompletenessValidator",
    "GuardrailValidator",
    "JsonSchemaValidator",
    "OnFailAction",
    "RegexValidator",
    "SchemaValidator",
    "ValidationResult",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single validation check.

    Attributes:
        valid: Whether the output passed validation.
        error_message: Human-readable description of the failure.
            ``None`` when ``valid`` is ``True``.
    """

    valid: bool
    error_message: str | None = None


class GuardrailValidator(abc.ABC):
    """Abstract base class for output validators."""

    @abc.abstractmethod
    def validate(self, output: str) -> ValidationResult:
        """Validate a raw LLM output string.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """


class JsonSchemaValidator(GuardrailValidator):
    """Validate that output is valid JSON conforming to a schema.

    Parameters:
        schema: A JSON Schema dict.  If ``None``, only syntactic
            JSON validity is checked.
        strict: When ``True`` (default), additional properties
            not defined in the schema cause validation failure.
    """

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self._schema = schema
        self._strict = strict
        self._effective_schema = (
            self._apply_strict(schema) if (schema is not None and strict) else schema
        )

    @property
    def schema(self) -> dict[str, Any] | None:
        """Return the JSON schema used for validation.

        Returns:
            The JSON Schema dict, or ``None`` if only syntax is
            checked.
        """
        return self._schema

    @staticmethod
    def _apply_strict(
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively add ``additionalProperties: false`` to objects.

        Creates a deep copy so the caller's original schema is
        not mutated.

        Parameters:
            schema: The original JSON schema dict.

        Returns:
            A new schema dict with strict enforcement applied.
        """

        def _enforce(node: dict[str, Any]) -> dict[str, Any]:
            # Treat nodes as objects if they explicitly declare
            # "type": "object" OR if they have "properties" (many
            # schemas omit the explicit type).
            is_object = node.get("type") == "object" or "properties" in node
            if is_object:
                node.setdefault("additionalProperties", False)
            # Recurse into properties
            for prop in node.get("properties", {}).values():
                if isinstance(prop, dict):
                    _enforce(prop)
            # Recurse into items (for arrays of objects)
            items = node.get("items")
            if isinstance(items, dict):
                _enforce(items)
            # Recurse into allOf / anyOf / oneOf
            for key in ("allOf", "anyOf", "oneOf"):
                for sub in node.get(key, []):
                    if isinstance(sub, dict):
                        _enforce(sub)
            return node

        return _enforce(copy.deepcopy(schema))

    def validate(self, output: str) -> ValidationResult:
        """Validate output as JSON, optionally against a schema.

        Uses ``json-repair`` to fix common LLM output issues such
        as markdown code fences, trailing commas, missing quotes,
        and truncated structures before validation.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """
        # Phase 1 — repair and parse JSON.
        # repair_json returns the repaired JSON as a string;
        # we then parse it ourselves so we can reject bare
        # scalars (plain text the repairer wrapped in quotes).
        repaired = repair_json(output)

        try:
            parsed = json.loads(repaired)
        except (json.JSONDecodeError, ValueError) as exc:
            return ValidationResult(
                valid=False,
                error_message=f"Invalid JSON: {exc}",
            )

        # Reject bare scalars — LLM output should be an
        # object or array, not a quoted string or number.
        if not isinstance(parsed, (dict, list)):
            return ValidationResult(
                valid=False,
                error_message=(
                    f"Expected a JSON object or array, got {type(parsed).__name__}"
                ),
            )

        # Phase 2 — schema check
        if self._effective_schema is not None:
            try:
                jsonschema.validate(
                    instance=parsed,
                    schema=self._effective_schema,
                )
            except jsonschema.ValidationError as exc:
                path = " -> ".join(str(p) for p in exc.absolute_path)
                return ValidationResult(
                    valid=False,
                    error_message=(
                        f"Schema validation failed at '{path}': {exc.message}"
                    ),
                )

        return ValidationResult(valid=True)


class RegexValidator(GuardrailValidator):
    """Validate that output matches a regular expression pattern.

    Parameters:
        pattern: A regex pattern string.  The output must contain
            at least one match.
        description: Human-readable description of what the
            pattern checks for, used in error messages.
    """

    def __init__(
        self,
        pattern: str,
        description: str = "output format",
    ) -> None:
        self._pattern = re.compile(pattern, re.DOTALL)
        self._description = description

    def validate(self, output: str) -> ValidationResult:
        """Validate that the output contains the expected pattern.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """
        if self._pattern.search(output):
            return ValidationResult(valid=True)
        return ValidationResult(
            valid=False,
            error_message=(
                f"Output does not match expected {self._description} pattern"
            ),
        )


# -------------------------------------------------------------------
# OnFailAction enum
# -------------------------------------------------------------------


class OnFailAction(enum.Enum):
    """Action to take when a validator fails.

    Attributes:
        EXCEPTION: Raise an exception immediately.
        REASK: Re-prompt the LLM with the validation error.
        FILTER: Silently discard the failing extraction.
        NOOP: Log the failure but take no corrective action.
    """

    EXCEPTION = "exception"
    REASK = "reask"
    FILTER = "filter"
    NOOP = "noop"


# -------------------------------------------------------------------
# SchemaValidator — Pydantic model validation
# -------------------------------------------------------------------


class SchemaValidator(GuardrailValidator):
    """Validate extraction output against a Pydantic model.

    Parses repaired JSON from the LLM output and validates each
    item against the provided Pydantic ``BaseModel`` subclass.
    Supports both single-object and array-of-objects outputs.

    Parameters:
        schema: A Pydantic ``BaseModel`` subclass.
        on_fail: Default failure action (stored for use by
            ``ValidatorChain``).  Does not affect the
            ``validate()`` return value itself.
        strict: When ``True``, Pydantic strict validation is
            used (no type coercion).
    """

    def __init__(
        self,
        schema: type,
        *,
        on_fail: OnFailAction = OnFailAction.REASK,
        strict: bool = False,
    ) -> None:
        # Import here to keep pydantic optional at module level
        from pydantic import BaseModel

        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                f"SchemaValidator requires a Pydantic BaseModel "
                f"subclass, got {schema!r}"
            )
        self._schema = schema
        self._on_fail = on_fail
        self._strict = strict

    @property
    def schema_class(self) -> type:
        """Return the Pydantic model class.

        Returns:
            The ``BaseModel`` subclass used for validation.
        """
        return self._schema

    @property
    def on_fail(self) -> OnFailAction:
        """Return the configured on-fail action.

        Returns:
            The ``OnFailAction`` value.
        """
        return self._on_fail

    def validate(self, output: str) -> ValidationResult:
        """Validate that the output conforms to the Pydantic schema.

        Attempts to repair common JSON issues, then validates each
        object in the output against the schema.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """
        from pydantic import ValidationError as PydanticValidationError

        repaired = repair_json(output)
        try:
            parsed = json.loads(repaired)
        except (json.JSONDecodeError, ValueError) as exc:
            return ValidationResult(
                valid=False,
                error_message=f"Invalid JSON: {exc}",
            )

        items = parsed if isinstance(parsed, list) else [parsed]

        errors: list[str] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(
                    f"Item {i}: expected object, got {type(item).__name__}"
                )
                continue
            try:
                self._schema.model_validate(
                    item, strict=self._strict
                )
            except PydanticValidationError as exc:
                for err in exc.errors():
                    loc = " -> ".join(str(l) for l in err["loc"])
                    errors.append(
                        f"Item {i}, field '{loc}': {err['msg']}"
                    )

        if errors:
            msg = (
                f"Pydantic validation failed for "
                f"{self._schema.__name__}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
            return ValidationResult(valid=False, error_message=msg)
        return ValidationResult(valid=True)


# -------------------------------------------------------------------
# ConfidenceThresholdValidator
# -------------------------------------------------------------------


class ConfidenceThresholdValidator(GuardrailValidator):
    """Reject extractions whose confidence falls below a threshold.

    In the LangCore pipeline, confidence scores are set **after**
    alignment — they do not appear in the raw LLM JSON output.
    This validator therefore provides two modes of operation:

    1. **Post-extraction mode** (:meth:`validate_extractions`) —
       operates on ``Extraction`` objects that already have
       ``confidence_score`` set.  This is the recommended mode.
    2. **Raw JSON mode** (:meth:`validate`) — inspects parsed JSON
       for a ``confidence_score`` key.  Useful when the LLM output
       already contains confidence values (e.g., custom pipelines).

    Parameters:
        min_confidence: Minimum acceptable confidence (0.0–1.0).
        score_key: The JSON key containing the confidence score.
            Defaults to ``"confidence_score"``.
        on_fail: Default failure action for ``ValidatorChain``.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        *,
        score_key: str = "confidence_score",
        on_fail: OnFailAction = OnFailAction.FILTER,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], "
                f"got {min_confidence}"
            )
        self._min_confidence = min_confidence
        self._score_key = score_key
        self._on_fail = on_fail

    @property
    def min_confidence(self) -> float:
        """Return the minimum confidence threshold.

        Returns:
            The configured minimum confidence value.
        """
        return self._min_confidence

    @property
    def on_fail(self) -> OnFailAction:
        """Return the configured on-fail action.

        Returns:
            The ``OnFailAction`` value.
        """
        return self._on_fail

    def validate(self, output: str) -> ValidationResult:
        """Validate that all extractions meet the confidence threshold.

        .. note::

            In the standard LangCore pipeline, confidence
            scores are set after alignment and will **not** appear
            in raw LLM JSON output.  If no items contain the
            ``score_key``, this validator passes by default.
            Use :meth:`validate_extractions` for post-alignment
            validation.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """
        repaired = repair_json(output)
        try:
            parsed = json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            # Not JSON — cannot check confidence; pass through
            # and let other validators handle JSON validity.
            return ValidationResult(valid=True)

        items = parsed if isinstance(parsed, list) else [parsed]
        below: list[str] = []

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            score = item.get(self._score_key)
            if score is not None:
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    below.append(
                        f"Item {i}: invalid confidence "
                        f"'{score}'"
                    )
                    continue
                if score_f < self._min_confidence:
                    below.append(
                        f"Item {i}: confidence {score_f:.4f} "
                        f"< threshold {self._min_confidence}"
                    )

        if below:
            return ValidationResult(
                valid=False,
                error_message=(
                    "Confidence threshold not met:\n"
                    + "\n".join(f"  - {b}" for b in below)
                ),
            )
        return ValidationResult(valid=True)

    def validate_extractions(
        self,
        extractions: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Filter extractions by confidence score (post-alignment).

        This is the recommended mode for checking confidence: it
        operates on ``Extraction`` objects whose
        ``confidence_score`` has been set by the alignment step.

        Parameters:
            extractions: A list of ``Extraction`` objects (or any
                object with a ``confidence_score`` attribute).

        Returns:
            A ``(passed, filtered)`` tuple of extraction lists.
        """
        passed: list[Any] = []
        filtered: list[Any] = []
        for ext in extractions:
            score = getattr(ext, "confidence_score", None)
            if score is not None and score < self._min_confidence:
                filtered.append(ext)
            else:
                passed.append(ext)
        return passed, filtered


# -------------------------------------------------------------------
# FieldCompletenessValidator
# -------------------------------------------------------------------


class FieldCompletenessValidator(GuardrailValidator):
    """Ensure all required Pydantic fields are present and non-empty.

    Goes beyond Pydantic's own ``required`` validation by also
    rejecting empty strings, empty lists, and ``None`` values for
    fields marked as required.

    Parameters:
        schema: A Pydantic ``BaseModel`` subclass whose required
            fields are enforced.
        on_fail: Default failure action for ``ValidatorChain``.
    """

    def __init__(
        self,
        schema: type,
        *,
        on_fail: OnFailAction = OnFailAction.REASK,
    ) -> None:
        from pydantic import BaseModel

        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                f"FieldCompletenessValidator requires a Pydantic "
                f"BaseModel subclass, got {schema!r}"
            )
        self._schema = schema
        self._on_fail = on_fail
        # Pre-compute required field names
        self._required_fields = self._get_required_fields(schema)

    @property
    def on_fail(self) -> OnFailAction:
        """Return the configured on-fail action.

        Returns:
            The ``OnFailAction`` value.
        """
        return self._on_fail

    @staticmethod
    def _get_required_fields(schema: type) -> set[str]:
        """Extract required field names from a Pydantic model.

        A field is considered required if it has no default value
        and is not ``Optional``.

        Parameters:
            schema: The Pydantic ``BaseModel`` subclass.

        Returns:
            A set of required field name strings.
        """
        required = set()
        for name, field_info in schema.model_fields.items():
            if field_info.is_required():
                required.add(name)
        return required

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """Check if a value is considered "empty".

        Parameters:
            value: The value to check.

        Returns:
            ``True`` if the value is ``None``, an empty string,
            or an empty collection.
        """
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict, set)) and len(value) == 0:
            return True
        return False

    def validate(self, output: str) -> ValidationResult:
        """Validate that required fields are present and non-empty.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """
        repaired = repair_json(output)
        try:
            parsed = json.loads(repaired)
        except (json.JSONDecodeError, ValueError) as exc:
            return ValidationResult(
                valid=False,
                error_message=f"Invalid JSON: {exc}",
            )

        items = parsed if isinstance(parsed, list) else [parsed]
        errors: list[str] = []

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(
                    f"Item {i}: expected object, "
                    f"got {type(item).__name__}"
                )
                continue
            for field_name in self._required_fields:
                if field_name not in item:
                    errors.append(
                        f"Item {i}: missing required field "
                        f"'{field_name}'"
                    )
                elif self._is_empty(item[field_name]):
                    errors.append(
                        f"Item {i}: required field "
                        f"'{field_name}' is empty"
                    )

        if errors:
            return ValidationResult(
                valid=False,
                error_message=(
                    f"Field completeness check failed for "
                    f"{self._schema.__name__}:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                ),
            )
        return ValidationResult(valid=True)


# -------------------------------------------------------------------
# ConsistencyValidator
# -------------------------------------------------------------------


class ConsistencyValidator(GuardrailValidator):
    """Cross-check extracted values using user-supplied rules.

    Each rule is a callable that receives a parsed dict and returns
    ``None`` on success or an error string on failure.  This
    allows domain-specific checks such as "start_date must be
    before end_date".

    Parameters:
        rules: A list of callables.  Each receives a ``dict`` and
            returns ``None`` (pass) or a ``str`` (error message).
        on_fail: Default failure action for ``ValidatorChain``.

    Example::

        def dates_ordered(data: dict) -> str | None:
            start = data.get("start_date", "")
            end = data.get("end_date", "")
            if start and end and start > end:
                return "start_date must be before end_date"
            return None

        validator = ConsistencyValidator(rules=[dates_ordered])
    """

    def __init__(
        self,
        rules: list[Callable[[dict[str, Any]], str | None]],
        *,
        on_fail: OnFailAction = OnFailAction.REASK,
    ) -> None:
        if not rules:
            raise ValueError(
                "ConsistencyValidator requires at least one rule"
            )
        self._rules = rules
        self._on_fail = on_fail

    @property
    def on_fail(self) -> OnFailAction:
        """Return the configured on-fail action.

        Returns:
            The ``OnFailAction`` value.
        """
        return self._on_fail

    def validate(self, output: str) -> ValidationResult:
        """Validate output against all consistency rules.

        Parameters:
            output: The raw text output from the language model.

        Returns:
            A ``ValidationResult`` indicating pass or fail.
        """
        repaired = repair_json(output)
        try:
            parsed = json.loads(repaired)
        except (json.JSONDecodeError, ValueError) as exc:
            return ValidationResult(
                valid=False,
                error_message=f"Invalid JSON: {exc}",
            )

        items = parsed if isinstance(parsed, list) else [parsed]
        errors: list[str] = []

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for rule in self._rules:
                error = rule(item)
                if error is not None:
                    errors.append(f"Item {i}: {error}")

        if errors:
            return ValidationResult(
                valid=False,
                error_message=(
                    "Consistency check failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                ),
            )
        return ValidationResult(valid=True)
