"""Validator registry and chain composition.

Provides a registry pattern for custom validators (inspired by
Guardrails Hub) and a ``ValidatorChain`` for composing multiple
validators with per-validator failure actions.

Example::

    from langextract_guardrails.validator_registry import (
        ValidatorChain,
        register_validator,
        get_validator,
    )

    @register_validator(name="my_check")
    class MyCheckValidator(GuardrailValidator):
        def validate(self, output: str) -> ValidationResult:
            ...

    chain = ValidatorChain([
        (SchemaValidator(MyModel), OnFailAction.REASK),
        (ConfidenceThresholdValidator(0.7), OnFailAction.FILTER),
    ])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langextract_guardrails.validators import (
    GuardrailValidator,
    OnFailAction,
    ValidationResult,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "ValidatorChain",
    "ValidatorEntry",
    "get_validator",
    "list_validators",
    "register_validator",
]

logger = logging.getLogger(__name__)

# Global registry: name -> validator class
_VALIDATOR_REGISTRY: dict[str, type[GuardrailValidator]] = {}


def register_validator(
    name: str,
) -> Any:
    """Decorator that registers a validator class by name.

    Parameters:
        name: A unique string identifier for the validator.

    Returns:
        A class decorator that registers the validator and
        returns the original class unchanged.

    Raises:
        ValueError: If ``name`` is already registered.

    Example::

        @register_validator(name="pii_check")
        class PiiValidator(GuardrailValidator):
            def validate(self, output: str) -> ValidationResult:
                ...
    """

    def _decorator(cls: type[GuardrailValidator]) -> type[GuardrailValidator]:
        if name in _VALIDATOR_REGISTRY:
            raise ValueError(
                f"Validator '{name}' is already registered "
                f"to {_VALIDATOR_REGISTRY[name].__name__}"
            )
        if not (isinstance(cls, type) and issubclass(cls, GuardrailValidator)):
            raise TypeError(
                f"@register_validator requires a GuardrailValidator "
                f"subclass, got {cls!r}"
            )
        _VALIDATOR_REGISTRY[name] = cls
        logger.debug("Registered validator: %s -> %s", name, cls.__name__)
        return cls

    return _decorator


def get_validator(name: str) -> type[GuardrailValidator]:
    """Look up a registered validator class by name.

    Parameters:
        name: The registered name.

    Returns:
        The validator class.

    Raises:
        KeyError: If no validator is registered under ``name``.
    """
    try:
        return _VALIDATOR_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_VALIDATOR_REGISTRY)) or "(none)"
        raise KeyError(
            f"No validator registered as '{name}'. " f"Available: {available}"
        ) from None


def list_validators() -> dict[str, type[GuardrailValidator]]:
    """Return a copy of all registered validators.

    Returns:
        A dict mapping registered names to validator classes.
    """
    return dict(_VALIDATOR_REGISTRY)


class ValidatorEntry:
    """Pairs a validator instance with its on-fail action.

    Parameters:
        validator: A ``GuardrailValidator`` instance.
        on_fail: What to do when this validator fails.  Defaults
            to ``OnFailAction.EXCEPTION``.
    """

    __slots__ = ("validator", "on_fail")

    def __init__(
        self,
        validator: GuardrailValidator,
        on_fail: OnFailAction = OnFailAction.EXCEPTION,
    ) -> None:
        self.validator = validator
        self.on_fail = on_fail

    def __repr__(self) -> str:
        return (
            f"ValidatorEntry({self.validator.__class__.__name__}, "
            f"on_fail={self.on_fail.value})"
        )


class ValidatorChain:
    """Compose multiple validators with per-validator failure actions.

    Each entry is a ``(validator, on_fail_action)`` pair.  When
    ``run()`` is called, validators execute in order.  The chain
    collects results and returns a ``ChainResult`` summarising the
    outcome.

    Parameters:
        entries: An iterable of ``ValidatorEntry`` objects, or
            ``(validator, on_fail_action)`` tuples that will be
            converted automatically.

    Example::

        chain = ValidatorChain([
            ValidatorEntry(SchemaValidator(Invoice), OnFailAction.REASK),
            ValidatorEntry(
                ConfidenceThresholdValidator(0.7),
                OnFailAction.FILTER,
            ),
        ])
        result = chain.run(output_text)
        if result.should_reask:
            # re-prompt the LLM
            ...
    """

    def __init__(
        self,
        entries: list[ValidatorEntry | tuple[GuardrailValidator, OnFailAction]],
    ) -> None:
        self._entries: list[ValidatorEntry] = []
        for entry in entries:
            if isinstance(entry, ValidatorEntry):
                self._entries.append(entry)
            elif isinstance(entry, tuple) and len(entry) == 2:
                self._entries.append(ValidatorEntry(entry[0], entry[1]))
            else:
                raise TypeError(
                    f"Expected ValidatorEntry or (validator, OnFailAction) "
                    f"tuple, got {type(entry)!r}"
                )

    @property
    def entries(self) -> list[ValidatorEntry]:
        """Return a copy of the validator entries.

        Returns:
            A list of ``ValidatorEntry`` objects.
        """
        return list(self._entries)

    def run(self, output: str) -> ChainResult:
        """Execute all validators against the output.

        Parameters:
            output: The raw LLM output string to validate.

        Returns:
            A ``ChainResult`` summarising the outcome.
        """
        failures: list[tuple[ValidatorEntry, ValidationResult]] = []
        should_reask = False
        should_filter = False

        for entry in self._entries:
            result = entry.validator.validate(output)
            if result.valid:
                continue

            failures.append((entry, result))
            if entry.on_fail == OnFailAction.REASK:
                should_reask = True
            elif entry.on_fail == OnFailAction.FILTER:
                should_filter = True
            elif entry.on_fail == OnFailAction.EXCEPTION:
                raise ValidationError(
                    f"{entry.validator.__class__.__name__} failed: "
                    f"{result.error_message}",
                    validator=entry.validator,
                    result=result,
                )
            # NOOP: record failure but take no action

        return ChainResult(
            passed=len(failures) == 0,
            failures=failures,
            should_reask=should_reask,
            should_filter=should_filter,
        )

    def as_guardrail_validators(self) -> list[GuardrailValidator]:
        """Extract the raw validator instances for use with
        ``GuardrailLanguageModel``.

        Returns:
            A list of ``GuardrailValidator`` instances.
        """
        return [e.validator for e in self._entries]


class ChainResult:
    """Result of running a ``ValidatorChain``.

    Attributes:
        passed: ``True`` if every validator passed.
        failures: List of ``(entry, result)`` for each failure.
        should_reask: ``True`` if any failure has
            ``OnFailAction.REASK``.
        should_filter: ``True`` if any failure has
            ``OnFailAction.FILTER``.
    """

    __slots__ = ("passed", "failures", "should_reask", "should_filter")

    def __init__(
        self,
        *,
        passed: bool,
        failures: list[tuple[ValidatorEntry, ValidationResult]],
        should_reask: bool,
        should_filter: bool,
    ) -> None:
        self.passed = passed
        self.failures = failures
        self.should_reask = should_reask
        self.should_filter = should_filter

    @property
    def error_messages(self) -> list[str]:
        """Collect all error messages from failures.

        Returns:
            A list of non-None error message strings.
        """
        return [
            r.error_message for _, r in self.failures if r.error_message is not None
        ]


class ValidationError(Exception):
    """Raised when a validator with ``OnFailAction.EXCEPTION`` fails.

    Attributes:
        validator: The failing validator instance.
        result: The ``ValidationResult`` that triggered the error.
    """

    def __init__(
        self,
        message: str,
        *,
        validator: GuardrailValidator,
        result: ValidationResult,
    ) -> None:
        super().__init__(message)
        self.validator = validator
        self.result = result
