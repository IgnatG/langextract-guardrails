"""LangCore guardrail provider plugin.

Wraps any ``BaseLanguageModel`` with output validation and
automatic retry with corrective prompts on validation failure.
Subsumes retry-optimisation and verification provider concepts.
"""

from langcore_guardrails.provider import GuardrailLanguageModel
from langcore_guardrails.validator_registry import (
    ChainResult,
    GuardrailValidationError,
    ValidationError,
    ValidatorChain,
    ValidatorEntry,
    get_validator,
    list_validators,
    register_validator,
)
from langcore_guardrails.validators import (
    ConfidenceThresholdValidator,
    ConsistencyValidator,
    FieldCompletenessValidator,
    GuardrailValidator,
    JsonSchemaValidator,
    OnFailAction,
    RegexValidator,
    SchemaValidator,
    ValidationResult,
)

__all__ = [
    "ChainResult",
    "ConfidenceThresholdValidator",
    "ConsistencyValidator",
    "FieldCompletenessValidator",
    "GuardrailLanguageModel",
    "GuardrailValidationError",
    "GuardrailValidator",
    "JsonSchemaValidator",
    "OnFailAction",
    "RegexValidator",
    "SchemaValidator",
    "ValidationError",
    "ValidationResult",
    "ValidatorChain",
    "ValidatorEntry",
    "get_validator",
    "list_validators",
    "register_validator",
]
__version__ = "1.2.0"
