"""Tests for GuardrailLanguageModel provider and validators."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from langextract_guardrails import (
    GuardrailLanguageModel,
    GuardrailValidator,
    JsonSchemaValidator,
    RegexValidator,
    ValidationResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ProgrammableProvider(BaseLanguageModel):
    """Provider that returns pre-configured responses per call.

    Each call to ``infer`` / ``async_infer`` pops the next response
    from the queue.  When the queue is exhausted, the last response
    is repeated.
    """

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        super().__init__()
        self._responses = list(responses)
        self._index = 0
        self.prompts_received: list[str] = []

    def _next_response(self) -> str:
        """Return the next configured response."""
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return text

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> Iterator[Sequence[ScoredOutput]]:
        for prompt in batch_prompts:
            self.prompts_received.append(prompt)
            text = self._next_response()
            yield [ScoredOutput(score=1.0, output=text)]

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> list[Sequence[ScoredOutput]]:
        results: list[Sequence[ScoredOutput]] = []
        for prompt in batch_prompts:
            self.prompts_received.append(prompt)
            text = self._next_response()
            results.append([ScoredOutput(score=1.0, output=text)])
        return results


class _AlwaysFailValidator(GuardrailValidator):
    """Validator that always fails with a fixed message."""

    def __init__(self, msg: str = "always fails") -> None:
        self._msg = msg

    def validate(self, output: str) -> ValidationResult:
        return ValidationResult(valid=False, error_message=self._msg)


class _AlwaysPassValidator(GuardrailValidator):
    """Validator that always passes."""

    def validate(self, output: str) -> ValidationResult:
        return ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestJsonSchemaValidator:
    """Tests for JsonSchemaValidator."""

    def test_valid_json_no_schema(self) -> None:
        v = JsonSchemaValidator()
        result = v.validate('{"key": "value"}')
        assert result.valid

    def test_invalid_json(self) -> None:
        v = JsonSchemaValidator()
        result = v.validate("not json at all")
        assert not result.valid
        assert result.error_message is not None

    def test_valid_json_with_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        v = JsonSchemaValidator(schema=schema)
        result = v.validate('{"name": "Alice", "age": 30}')
        assert result.valid

    def test_invalid_json_with_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        v = JsonSchemaValidator(schema=schema)
        result = v.validate('{"age": 30}')
        assert not result.valid
        assert "Schema validation failed" in (result.error_message or "")

    def test_strips_markdown_fences(self) -> None:
        v = JsonSchemaValidator()
        fenced = '```json\n{"key": "value"}\n```'
        result = v.validate(fenced)
        assert result.valid

    def test_strips_plain_fences(self) -> None:
        v = JsonSchemaValidator()
        fenced = '```\n{"key": "value"}\n```'
        result = v.validate(fenced)
        assert result.valid


class TestRegexValidator:
    """Tests for RegexValidator."""

    def test_matching_output(self) -> None:
        v = RegexValidator(r'"name"\s*:', description="JSON name field")
        result = v.validate('{"name": "Alice"}')
        assert result.valid

    def test_non_matching_output(self) -> None:
        v = RegexValidator(r'"name"\s*:', description="JSON name field")
        result = v.validate("no name here")
        assert not result.valid
        assert "JSON name field" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Provider tests — sync
# ---------------------------------------------------------------------------


class TestGuardrailProviderSync:
    """Tests for GuardrailLanguageModel sync inference."""

    def test_passes_valid_output_directly(self) -> None:
        inner = _ProgrammableProvider(responses=['{"name": "Alice"}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
        )
        results = list(guard.infer(["extract name"]))
        assert len(results) == 1
        assert results[0][0].score == 1.0
        # Only one call — no retries needed
        assert len(inner.prompts_received) == 1

    def test_retries_on_invalid_output(self) -> None:
        inner = _ProgrammableProvider(
            responses=[
                "not json",  # attempt 1 — fail
                '{"name": "Alice"}',  # attempt 2 — pass
            ]
        )
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=3,
        )
        results = list(guard.infer(["extract name"]))
        assert len(results) == 1
        assert results[0][0].score == 1.0
        assert results[0][0].output == '{"name": "Alice"}'
        # Two calls: original + one retry
        assert len(inner.prompts_received) == 2

    def test_correction_prompt_includes_error(self) -> None:
        inner = _ProgrammableProvider(
            responses=[
                "bad output",
                '{"valid": true}',
            ]
        )
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=1,
        )
        list(guard.infer(["original prompt"]))
        # Second prompt should contain the error
        correction = inner.prompts_received[1]
        assert "original prompt" in correction
        assert "bad output" in correction
        assert "Invalid JSON" in correction

    def test_exhausts_retries_returns_score_zero(self) -> None:
        inner = _ProgrammableProvider(responses=["not json"] * 5)
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=2,
        )
        results = list(guard.infer(["prompt"]))
        assert results[0][0].score == 0.0
        # 1 initial + 2 retries = 3 calls
        assert len(inner.prompts_received) == 3

    def test_multiple_validators_all_must_pass(self) -> None:
        inner = _ProgrammableProvider(responses=['{"name": "Alice"}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[
                JsonSchemaValidator(),
                RegexValidator(r'"name"'),
            ],
        )
        results = list(guard.infer(["prompt"]))
        assert results[0][0].score == 1.0

    def test_first_failing_validator_triggers_retry(self) -> None:
        inner = _ProgrammableProvider(
            responses=[
                '{"foo": "bar"}',  # valid JSON, no "name"
                '{"name": "Alice"}',  # valid JSON, has "name"
            ]
        )
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[
                JsonSchemaValidator(),  # passes
                RegexValidator(r'"name"', "name key"),  # fails 1st
            ],
            max_retries=2,
        )
        results = list(guard.infer(["prompt"]))
        assert results[0][0].output == '{"name": "Alice"}'
        assert len(inner.prompts_received) == 2

    def test_batch_prompts_independent(self) -> None:
        inner = _ProgrammableProvider(
            responses=[
                '{"a": 1}',  # prompt 1 — pass
                "invalid",  # prompt 2, attempt 1 — fail
                '{"b": 2}',  # prompt 2, attempt 2 — pass
            ]
        )
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=2,
        )
        results = list(guard.infer(["p1", "p2"]))
        assert len(results) == 2
        assert results[0][0].output == '{"a": 1}'
        assert results[1][0].output == '{"b": 2}'

    def test_kwargs_forwarded_to_inner(self) -> None:
        inner = _ProgrammableProvider(responses=['{"ok": true}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
        )
        with mock.patch.object(inner, "infer", wraps=inner.infer) as m:
            list(guard.infer(["prompt"], pass_num=2))
        m.assert_called_with(mock.ANY, pass_num=2)

    def test_custom_correction_template(self) -> None:
        inner = _ProgrammableProvider(responses=["bad", '{"ok": true}'])
        template = "CUSTOM: {original_prompt} | {invalid_output} | {error_message}"
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            correction_template=template,
        )
        list(guard.infer(["my prompt"]))
        correction = inner.prompts_received[1]
        assert correction.startswith("CUSTOM: my prompt")

    def test_include_output_in_correction_true_by_default(self) -> None:
        """Correction prompt includes the invalid output by default."""
        inner = _ProgrammableProvider(responses=["bad output", '{"ok": true}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
        )
        list(guard.infer(["prompt"]))
        correction = inner.prompts_received[1]
        assert "bad output" in correction

    def test_error_only_correction_omits_invalid_output(self) -> None:
        """When include_output_in_correction=False, the invalid output
        must not appear in the correction prompt, reducing token usage
        and avoiding the model fixating on junk output."""
        inner = _ProgrammableProvider(responses=["BAD OUTPUT TEXT", '{"ok": true}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            include_output_in_correction=False,
        )
        list(guard.infer(["original prompt"]))
        correction = inner.prompts_received[1]
        assert "BAD OUTPUT TEXT" not in correction
        # Original prompt and error still present.
        assert "original prompt" in correction

    def test_error_only_mode_still_corrects_successfully(self) -> None:
        """The provider should still succeed after a retry in error-only mode."""
        inner = _ProgrammableProvider(responses=["not json", '{"value": 42}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            include_output_in_correction=False,
        )
        results = list(guard.infer(["prompt"]))
        assert results[0][0].output == '{"value": 42}'
        assert results[0][0].score == 1.0


# ---------------------------------------------------------------------------
# Provider tests — async
# ---------------------------------------------------------------------------


class TestGuardrailProviderAsync:
    """Tests for GuardrailLanguageModel async inference."""

    @pytest.mark.asyncio
    async def test_async_passes_valid(self) -> None:
        inner = _ProgrammableProvider(responses=['{"name": "Bob"}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
        )
        results = await guard.async_infer(["prompt"])
        assert len(results) == 1
        assert results[0][0].score == 1.0

    @pytest.mark.asyncio
    async def test_async_retries_on_failure(self) -> None:
        inner = _ProgrammableProvider(responses=["bad", '{"ok": true}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=2,
        )
        results = await guard.async_infer(["prompt"])
        assert results[0][0].output == '{"ok": true}'
        assert len(inner.prompts_received) == 2

    @pytest.mark.asyncio
    async def test_async_exhausts_retries(self) -> None:
        inner = _ProgrammableProvider(responses=["bad"] * 5)
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=1,
        )
        results = await guard.async_infer(["prompt"])
        assert results[0][0].score == 0.0


# ---------------------------------------------------------------------------
# Plugin registration test
# ---------------------------------------------------------------------------


class TestJsonSchemaValidatorStrict:
    """Tests for JsonSchemaValidator strict mode."""

    def test_strict_rejects_additional_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        v = JsonSchemaValidator(schema=schema, strict=True)
        result = v.validate('{"name": "Alice", "extra": "bad"}')
        assert not result.valid
        assert "additional" in (result.error_message or "").lower()

    def test_non_strict_allows_additional_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        v = JsonSchemaValidator(schema=schema, strict=False)
        result = v.validate('{"name": "Alice", "extra": "ok"}')
        assert result.valid

    def test_strict_recursive_nested_objects(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
            },
        }
        v = JsonSchemaValidator(schema=schema, strict=True)
        result = v.validate('{"data": {"value": 1, "extra": 2}}')
        assert not result.valid


class TestCorrectionTruncation:
    """Tests for correction prompt truncation."""

    def test_truncation_applied_to_correction_prompt(self) -> None:
        inner = _ProgrammableProvider(responses=["x" * 500, '{"ok": true}'])
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[JsonSchemaValidator()],
            max_retries=1,
            max_correction_prompt_length=20,
            max_correction_output_length=10,
        )
        list(guard.infer(["A" * 200]))
        correction = inner.prompts_received[1]
        # Original prompt should be truncated
        assert "A" * 20 + "..." in correction
        # Invalid output should be truncated
        assert "x" * 10 + "..." in correction


class TestPickBest:
    """Tests for best-by-score selection."""

    def test_picks_highest_scored_output(self) -> None:
        """Provider should validate the highest-scored output."""

        class _MultiOutputProvider(BaseLanguageModel):
            def infer(self, batch_prompts, **kw):
                for _p in batch_prompts:
                    yield [
                        ScoredOutput(score=0.3, output="low"),
                        ScoredOutput(
                            score=0.9,
                            output='{"best": true}',
                        ),
                        ScoredOutput(score=0.5, output="mid"),
                    ]

            async def async_infer(self, batch_prompts, **kw):
                return [
                    [
                        ScoredOutput(score=0.3, output="low"),
                        ScoredOutput(
                            score=0.9,
                            output='{"best": true}',
                        ),
                    ]
                ]

        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=_MultiOutputProvider(),
            validators=[JsonSchemaValidator()],
        )
        results = list(guard.infer(["prompt"]))
        # Should have validated the score=0.9 output
        assert results[0][0].output == '{"best": true}'


# ---------------------------------------------------------------------------
# Plugin registration test
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """Tests for entry-point discovery."""

    def test_guardrails_prefix_resolves(self) -> None:
        import langextract as lx
        from langextract.providers import registry

        lx.providers.load_plugins_once()
        cls = registry.resolve("guardrails/my-model")
        assert cls.__name__ == "GuardrailLanguageModel"
