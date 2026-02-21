"""Pluggable validators for the guardrail provider.

Each validator inspects a raw LLM output string and returns a
``ValidationResult`` indicating whether the output is acceptable.
When validation fails, the error message is injected into a
corrective prompt for retry.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import jsonschema
from json_repair import repair_json

__all__ = [
    "GuardrailValidator",
    "JsonSchemaValidator",
    "RegexValidator",
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
        import copy

        def _enforce(node: dict[str, Any]) -> dict[str, Any]:
            if node.get("type") == "object":
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
