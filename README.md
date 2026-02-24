# LangCore Guardrails

> Provider plugin for [LangCore](https://github.com/ignatg/langcore) — output validation, automatic retry with corrective prompts, and a rich set of pluggable validators.

[![PyPI version](https://img.shields.io/pypi/v/langcore-guardrails)](https://pypi.org/project/langcore-guardrails/)
[![Python](https://img.shields.io/pypi/pyversions/langcore-guardrails)](https://pypi.org/project/langcore-guardrails/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

**langcore-guardrails** is a provider plugin for [LangCore](https://github.com/ignatg/langcore) that validates LLM output against schemas, confidence thresholds, regex patterns, and custom business rules — automatically retrying with corrective prompts when validation fails. It wraps any `BaseLanguageModel` and acts as a self-correcting layer between the LLM and your application.

---

## Features

- **Corrective retry loop** — when validation fails, constructs a corrective prompt with the original request, invalid output, and error details, then re-prompts the LLM (up to `max_retries`)
- **7 built-in validators**:
  - `JsonSchemaValidator` — validates output against a JSON Schema with auto-repair and markdown fence stripping
  - `RegexValidator` — matches output against regular expression patterns
  - `SchemaValidator` — validates against Pydantic `BaseModel` classes (strict or lenient mode)
  - `ConfidenceThresholdValidator` — rejects extractions below a confidence score threshold
  - `FieldCompletenessValidator` — ensures required fields are present and non-empty
  - `ConsistencyValidator` — cross-checks extracted values using custom business rules
  - `GroundingValidator` — rejects hallucinated or poorly-grounded extractions using LangCore's alignment engine
- **4 on-fail actions** — `EXCEPTION` (raise immediately), `REASK` (re-prompt LLM), `FILTER` (silently discard), `NOOP` (log and continue)
- **Validator chaining** — compose multiple validators with per-validator failure actions via `ValidatorChain`
- **Validator registry** — register custom validators by name with `@register_validator` for discovery and reuse
- **Error-only correction mode** — omit invalid output from retry prompts to save tokens on large payloads
- **Correction prompt truncation** — bound prompt and output length in corrective prompts
- **Custom correction templates** — fully customizable retry prompt format
- **Markdown fence stripping** — automatic via `json-repair` for LLMs that wrap JSON in code fences
- **Batch-independent retries** — each prompt in a batch retries independently
- **Async concurrency control** — `max_concurrency` semaphore for async batches
- **Zero-config plugin** — auto-registered via Python entry points

---

## Installation

```bash
pip install langcore-guardrails
```

---

## Quick Start

### Integration with LangCore

langcore-guardrails integrates with LangCore through the **decorator provider pattern**. Wrap any LangCore model to add output validation:

```python
import langcore as lx
from pydantic import BaseModel, Field
from langcore_guardrails import (
    GuardrailLanguageModel,
    SchemaValidator,
    ConfidenceThresholdValidator,
    OnFailAction,
)

# Define expected output schema
class ContractEntity(BaseModel):
    party_name: str = Field(description="Name of the contracting party")
    role: str = Field(description="Role in the contract (buyer, seller, etc.)")
    effective_date: str = Field(description="Contract effective date (YYYY-MM-DD)")

# Create the base LLM provider
inner_model = lx.factory.create_model(
    lx.factory.ModelConfig(model_id="litellm/gpt-4o", provider="LiteLLMLanguageModel")
)

# Wrap with guardrails
guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[
        SchemaValidator(ContractEntity, on_fail=OnFailAction.REASK),
        ConfidenceThresholdValidator(min_confidence=0.7, on_fail=OnFailAction.FILTER),
    ],
    max_retries=3,
)

# Use as a drop-in replacement — validation and retries happen automatically
result = lx.extract(
    text_or_documents="This Agreement is entered into by Acme Corp (Seller) and Beta LLC (Buyer), effective January 15, 2025.",
    model=guard_model,
    prompt_description="Extract contracting parties with roles and dates.",
    examples=[
        lx.data.ExampleData(
            text="Agreement between Alpha Inc (Lessor) and Omega Ltd (Lessee), effective March 1, 2024.",
            extractions=[
                lx.data.Extraction("party", "Alpha Inc", attributes={"role": "Lessor", "effective_date": "2024-03-01"}),
                lx.data.Extraction("party", "Omega Ltd", attributes={"role": "Lessee", "effective_date": "2024-03-01"}),
            ],
        )
    ],
)
```

---

## Built-in Validators

### JsonSchemaValidator

Validates LLM output against a JSON Schema. Automatically strips markdown fences and repairs common JSON issues:

```python
from langcore_guardrails import JsonSchemaValidator

schema = {
    "type": "object",
    "properties": {
        "parties": {"type": "array", "items": {"type": "string"}},
        "effective_date": {"type": "string"},
    },
    "required": ["parties", "effective_date"],
}

validator = JsonSchemaValidator(schema=schema, strict=True)
```

### RegexValidator

Validates that output matches a regular expression pattern:

```python
from langcore_guardrails import RegexValidator

validator = RegexValidator(r'\d{4}-\d{2}-\d{2}', description="ISO date format")
```

### SchemaValidator

Validates extraction output against a Pydantic `BaseModel`. Supports strict mode (no type coercion) and lenient mode:

```python
from pydantic import BaseModel, Field
from langcore_guardrails import SchemaValidator, OnFailAction

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    amount: float = Field(description="Total amount in USD")
    due_date: str = Field(description="Due date (YYYY-MM-DD)")

validator = SchemaValidator(
    Invoice,
    on_fail=OnFailAction.REASK,  # Re-prompt LLM on failure
    strict=False,                 # Allow type coercion (e.g., "100" → 100.0)
)
```

### ConfidenceThresholdValidator

Rejects extractions whose `confidence_score` falls below a threshold. Works with LangCore's built-in multi-pass confidence scoring:

```python
from langcore_guardrails import ConfidenceThresholdValidator, OnFailAction

validator = ConfidenceThresholdValidator(
    min_confidence=0.7,
    score_key="confidence_score",
    on_fail=OnFailAction.FILTER,  # Silently discard low-confidence results
)
```

### FieldCompletenessValidator

Ensures all required Pydantic fields are present **and** non-empty (rejects empty strings, empty lists, and `None`):

```python
from langcore_guardrails import FieldCompletenessValidator

validator = FieldCompletenessValidator(Invoice, on_fail=OnFailAction.REASK)
```

### ConsistencyValidator

Cross-checks extracted values using user-supplied business rules. Each rule returns `None` on success or an error string on failure:

```python
from langcore_guardrails import ConsistencyValidator

def dates_ordered(data: dict) -> str | None:
    start = data.get("start_date", "")
    end = data.get("end_date", "")
    if start and end and start > end:
        return "start_date must be before end_date"
    return None

def positive_amount(data: dict) -> str | None:
    if data.get("amount", 0) < 0:
        return "amount must be non-negative"
    return None

validator = ConsistencyValidator(
    rules=[dates_ordered, positive_amount],
    on_fail=OnFailAction.REASK,
)
```

### GroundingValidator

Rejects extractions that are not grounded in the source text. Works **post-alignment** — uses LangCore's alignment engine signals (`alignment_status` and `char_interval`) to detect hallucinated or weakly-grounded extractions:

- **Unaligned** — no `alignment_status` (hallucinated text not found in source)
- **Below minimum alignment quality** — e.g. only fuzzy-matched when `MATCH_LESSER` or better is required
- **Below minimum character coverage** — the overlapping span in the source text is too short relative to the extraction

```python
from langcore_guardrails import GroundingValidator, OnFailAction

validator = GroundingValidator(
    min_alignment_quality="MATCH_FUZZY",  # Accepts all aligned, rejects unaligned
    min_coverage=0.5,                      # At least 50% character overlap
    on_fail=OnFailAction.FILTER,           # Silently remove hallucinated extractions
)

# Use after alignment (post-extraction)
passed, filtered = validator.validate_extractions(
    extractions=result.extractions,
    source_text=result.text,
)

# Or convenience method for AnnotatedDocument
passed, filtered = validator.validate_document(result)
```

**Alignment quality levels** (best → worst):

| Level | Meaning |
|-------|--------|
| `MATCH_EXACT` | Extraction text found verbatim in source |
| `MATCH_GREATER` | Source span is larger than extraction text |
| `MATCH_LESSER` | Source span is smaller than extraction text |
| `MATCH_FUZZY` | Approximate match only |

> **Note:** `GroundingValidator` is a post-alignment validator. The `validate()` method (raw LLM output) always passes. Use `validate_extractions()` or `validate_document()` after LangCore's alignment step for meaningful grounding checks.

---

## On-Fail Actions

The `OnFailAction` enum controls what happens when a validator fails:

| Action | Behaviour |
|--------|-----------|
| `EXCEPTION` | Raise a `GuardrailValidationError` immediately |
| `REASK` | Re-prompt the LLM with the validation error as feedback |
| `FILTER` | Silently discard the failing output |
| `NOOP` | Log the failure but return the output as-is |

---

## Validator Chaining

Compose multiple validators with per-validator failure actions using `ValidatorChain`:

```python
from langcore_guardrails import (
    ValidatorChain,
    ValidatorEntry,
    SchemaValidator,
    ConfidenceThresholdValidator,
    FieldCompletenessValidator,
    OnFailAction,
)

chain = ValidatorChain([
    ValidatorEntry(SchemaValidator(Invoice), OnFailAction.REASK),
    ValidatorEntry(FieldCompletenessValidator(Invoice), OnFailAction.REASK),
    ValidatorEntry(
        ConfidenceThresholdValidator(min_confidence=0.7),
        OnFailAction.FILTER,
    ),
])

result = chain.run(llm_output)
if result.should_reask:
    print("Re-prompting LLM:", result.error_messages)
elif result.should_filter:
    print("Filtering low-confidence extractions")
```

---

## Custom Validators

### Implementing a Validator

Create custom validators by implementing the `GuardrailValidator` interface:

```python
from langcore_guardrails import GuardrailValidator, ValidationResult

class MaxLengthValidator(GuardrailValidator):
    def __init__(self, max_chars: int) -> None:
        self._max = max_chars

    def validate(self, output: str) -> ValidationResult:
        if len(output) <= self._max:
            return ValidationResult(valid=True)
        return ValidationResult(
            valid=False,
            error_message=f"Output exceeds {self._max} characters ({len(output)} chars)",
        )
```

### Registering for Discovery

Register validators by name for dynamic lookup:

```python
from langcore_guardrails import register_validator, get_validator

@register_validator(name="max_length")
class MaxLengthValidator(GuardrailValidator):
    def __init__(self, max_chars: int = 5000) -> None:
        self._max = max_chars

    def validate(self, output: str) -> ValidationResult:
        if len(output) <= self._max:
            return ValidationResult(valid=True)
        return ValidationResult(
            valid=False,
            error_message=f"Output exceeds {self._max} chars ({len(output)})",
        )

# Retrieve by name anywhere in your codebase
cls = get_validator("max_length")
validator = cls(max_chars=10000)
```

---

## Advanced Configuration

### Error-Only Correction Mode

Omit the invalid output from the correction prompt to save tokens on large payloads:

```python
guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[SchemaValidator(Invoice)],
    include_output_in_correction=False,
    max_retries=3,
)
```

### Correction Prompt Truncation

Limit the size of corrective prompts to control token costs:

```python
guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[JsonSchemaValidator()],
    max_correction_prompt_length=2000,
    max_correction_output_length=1000,
)
```

### Custom Correction Template

Provide your own template for corrective retry prompts:

```python
template = (
    "The output was invalid.\n"
    "Original request: {original_prompt}\n"
    "Your response: {invalid_output}\n"
    "Error: {error_message}\n"
    "Please fix the output."
)

guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[JsonSchemaValidator()],
    correction_template=template,
)
```

### Async Usage

Each prompt in a batch retries independently:

```python
results = await guard_model.async_infer(["prompt1", "prompt2"])
```

---

## How It Works

1. The original prompt is sent to the inner provider
2. The response is validated against all configured validators (in order)
3. If validation passes, the result is returned unchanged
4. If validation fails:
   - A corrective prompt is constructed with the original prompt, invalid output, and error message
   - The corrective prompt is sent to the inner provider
   - Steps 2–4 repeat up to `max_retries` times
5. If all retries are exhausted, the last result is returned with `score=0.0`

---

## Composing with Other Plugins

langcore-guardrails composes naturally with other LangCore decorator providers:

```python
import langcore as lx
from langcore_audit import AuditLanguageModel, JsonFileSink
from langcore_guardrails import GuardrailLanguageModel, SchemaValidator, OnFailAction
from langcore_hybrid import HybridLanguageModel, RegexRule, RuleConfig

# Base provider
llm = lx.factory.create_model(
    lx.factory.ModelConfig(model_id="litellm/gpt-4o", provider="LiteLLMLanguageModel")
)

# Layer 1: Rule-based extraction for known patterns
hybrid = HybridLanguageModel(
    model_id="hybrid/gpt-4o", inner=llm,
    rule_config=RuleConfig(rules=[RegexRule(r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})")]),
)

# Layer 2: Output validation
guarded = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o", inner=hybrid,
    validators=[SchemaValidator(MySchema, on_fail=OnFailAction.REASK)],
    max_retries=3,
)

# Layer 3: Audit logging
audited = AuditLanguageModel(
    model_id="audit/gpt-4o", inner=guarded,
    sinks=[JsonFileSink("./audit.jsonl")],
)

result = lx.extract(text_or_documents="...", model=audited, ...)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python ≥ 3.12
- `langcore`
- `jsonschema` ≥ 4.26.0
- `json-repair` ≥ 0.58.0
- `pydantic` ≥ 2.12.5

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
