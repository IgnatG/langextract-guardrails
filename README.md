# LangCore Guardrails Provider

A provider plugin for [LangCore](https://github.com/google/langcore) that wraps any `BaseLanguageModel` with output validation, automatic retry with corrective prompts, and a rich set of pluggable validators. Inspired by [Instructor](https://github.com/jxnl/instructor) and [Guardrails AI](https://github.com/guardrails-ai/guardrails).

> **Note**: This is a third-party provider plugin for LangCore. For the main LangCore library, visit [google/langcore](https://github.com/google/langcore).

## Installation

Install from source:

```bash
git clone <repo-url>
cd langcore-guardrails
pip install -e .
```

## Features at a Glance

| Feature | langcore-guardrails | Instructor | Guardrails AI |
|---|---|---|---|
| **Validation + retry loop** | ✅ Corrective prompts with error feedback | ✅ Automatic retry on Pydantic failure | ✅ Guard wrapping with retry |
| **Pydantic schema validation** | ✅ `SchemaValidator` — strict or coercive | ✅ Native Pydantic response model | ⚠️ Via Pydantic integration |
| **JSON Schema validation** | ✅ `JsonSchemaValidator` with strict mode | ❌ | ✅ JSON schema guard |
| **Confidence threshold** | ✅ `ConfidenceThresholdValidator` — per-extraction | ❌ | ❌ |
| **Field completeness** | ✅ `FieldCompletenessValidator` — empty/None checks | ❌ | ⚠️ Via custom validators |
| **Consistency rules** | ✅ `ConsistencyValidator` — user-supplied rules | ❌ | ⚠️ Via custom validators |
| **Regex validation** | ✅ `RegexValidator` | ❌ | ✅ Regex guard |
| **On-fail actions** | ✅ `EXCEPTION` / `REASK` / `FILTER` / `NOOP` | ⚠️ Exception only | ✅ `EXCEPTION` / `REASK` / `FIX` / `NOOP` |
| **Validator registry** | ✅ `@register_validator` decorator | ❌ | ✅ Guardrails Hub (67+ validators) |
| **Validator chaining** | ✅ `ValidatorChain` with per-validator actions | ❌ | ✅ Guard chaining |
| **Error-only correction mode** | ✅ Omit invalid output from retry prompt | ❌ | ❌ |
| **Correction prompt truncation** | ✅ Bounded prompt/output length | ❌ | ❌ |
| **Markdown fence stripping** | ✅ Via `json-repair` | ❌ | ❌ |
| **Batch-independent retries** | ✅ Each prompt retries independently | ❌ | ❌ |
| **Async concurrency control** | ✅ `max_concurrency` semaphore | ✅ | ❌ |
| **LangCore integration** | ✅ Native `BaseLanguageModel` provider | ❌ (OpenAI-focused) | ❌ (LLM-agnostic but no LangCore) |

## Built-in Validators

### `JsonSchemaValidator`

Validates that LLM output is valid JSON conforming to a JSON Schema. Automatically strips markdown fences and repairs common issues.

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

### `RegexValidator`

Validates that output matches a regular expression pattern.

```python
from langcore_guardrails import RegexValidator

validator = RegexValidator(
    r'\d{4}-\d{2}-\d{2}',
    description="date format"
)
```

### `SchemaValidator`

Validates extraction output against a Pydantic `BaseModel`. Supports strict mode (no type coercion) and lenient mode.

```python
from pydantic import BaseModel, Field
from langcore_guardrails import SchemaValidator, OnFailAction

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    amount: float = Field(description="Total amount")
    due_date: str = Field(description="Due date YYYY-MM-DD")

validator = SchemaValidator(
    Invoice,
    on_fail=OnFailAction.REASK,  # re-prompt LLM on failure
    strict=False,                 # allow type coercion
)
```

### `ConfidenceThresholdValidator`

Rejects extractions whose `confidence_score` falls below a threshold. Works with LangCore's built-in confidence scoring.

```python
from langcore_guardrails import ConfidenceThresholdValidator, OnFailAction

validator = ConfidenceThresholdValidator(
    min_confidence=0.7,
    score_key="confidence_score",  # default
    on_fail=OnFailAction.FILTER,   # silently discard low-confidence
)
```

### `FieldCompletenessValidator`

Ensures all required Pydantic fields are present *and* non-empty (rejects empty strings, empty lists, and `None` values).

```python
from langcore_guardrails import FieldCompletenessValidator

validator = FieldCompletenessValidator(
    Invoice,
    on_fail=OnFailAction.REASK,
)
```

### `ConsistencyValidator`

Cross-checks extracted values using user-supplied rules. Each rule is a callable that returns `None` on success or an error string on failure.

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

## On-Fail Actions

The `OnFailAction` enum controls what happens when a validator fails:

| Action | Behaviour |
|---|---|
| `EXCEPTION` | Raise a `ValidationError` immediately |
| `REASK` | Re-prompt the LLM with the validation error |
| `FILTER` | Silently discard the failing output |
| `NOOP` | Log the failure but take no action |

## Validator Registry

Register custom validators by name for discovery and reuse:

```python
from langcore_guardrails import register_validator, get_validator, GuardrailValidator, ValidationResult

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

# Later, retrieve by name:
cls = get_validator("max_length")
validator = cls(max_chars=10000)
```

## Validator Chaining

Compose multiple validators with per-validator failure actions:

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

## Usage with GuardrailLanguageModel

### Basic: Pydantic Schema Validation with Retry

```python
import langcore as lx
from langcore_guardrails import (
    GuardrailLanguageModel,
    SchemaValidator,
    ConfidenceThresholdValidator,
    OnFailAction,
)

inner_model = lx.factory.create_model(
    lx.factory.ModelConfig(model_id="litellm/azure/gpt-4o", provider="LiteLLMLanguageModel")
)

guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[
        SchemaValidator(Invoice, on_fail=OnFailAction.REASK),
        ConfidenceThresholdValidator(min_confidence=0.7, on_fail=OnFailAction.FILTER),
    ],
    max_retries=3,
)

result = lx.extract(
    text_or_documents="Invoice INV-2024-789 for $3,450 is due April 20th, 2024",
    model=guard_model,
    prompt_description="Extract invoice data as JSON.",
)
```

### JSON Schema Validation

```python
from langcore_guardrails import GuardrailLanguageModel, JsonSchemaValidator

guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[JsonSchemaValidator(schema=my_json_schema)],
    max_retries=3,
)
```

### Combining Multiple Validators

Validators are applied in order. The first failure triggers a retry:

```python
from langcore_guardrails import (
    GuardrailLanguageModel,
    SchemaValidator,
    FieldCompletenessValidator,
    ConsistencyValidator,
    ConfidenceThresholdValidator,
)

guard_model = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=inner_model,
    validators=[
        SchemaValidator(Invoice),
        FieldCompletenessValidator(Invoice),
        ConsistencyValidator(rules=[dates_ordered]),
        ConfidenceThresholdValidator(min_confidence=0.7),
    ],
    max_retries=3,
)
```

### Error-Only Correction Mode

When the invalid output is long, omit it from the correction prompt to save tokens:

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

For large prompts or outputs, truncate what goes into the correction prompt:

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

### Custom Validators

Implement the `GuardrailValidator` interface:

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

### Async Usage

```python
results = await guard_model.async_infer(["prompt1", "prompt2"])
# Each prompt independently validates and retries
```

## How It Works

1. The original prompt is sent to the inner provider
2. The response is validated against all configured validators (in order)
3. If validation passes, the result is returned as-is
4. If validation fails:
   a. A corrective prompt is constructed with the original prompt, invalid output, and error message
   b. The corrective prompt is sent to the inner provider
   c. Steps 2-4 repeat up to `max_retries` times
5. If all retries are exhausted, the last result is returned with `score=0.0`

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
