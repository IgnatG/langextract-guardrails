"""LangExtract guardrail provider plugin.

Wraps any ``BaseLanguageModel`` with output validation and
automatic retry with corrective prompts on validation failure.
Subsumes retry-optimisation and verification provider concepts.
"""

from langextract_guardrails.provider import GuardrailLanguageModel
from langextract_guardrails.validators import (
    GuardrailValidator,
    JsonSchemaValidator,
    RegexValidator,
    ValidationResult,
)

__all__ = [
    "GuardrailLanguageModel",
    "GuardrailValidator",
    "JsonSchemaValidator",
    "RegexValidator",
    "ValidationResult",
]
__version__ = "1.0.0"
