"""Tests for Phase 4 validators and registry.

Tests new validators (SchemaValidator, ConfidenceThresholdValidator,
FieldCompletenessValidator, ConsistencyValidator), the OnFailAction
enum, the validator registry, and the ValidatorChain.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from langcore_guardrails import (
    ChainResult,
    ConfidenceThresholdValidator,
    ConsistencyValidator,
    FieldCompletenessValidator,
    GuardrailValidationError,
    GuardrailValidator,
    OnFailAction,
    SchemaValidator,
    ValidationResult,
    ValidatorChain,
    ValidatorEntry,
    get_validator,
    list_validators,
    register_validator,
)

# ---------------------------------------------------------------------------
# Test Pydantic models
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    """Sample schema for testing."""

    invoice_number: str = Field(description="Invoice ID")
    amount: float = Field(description="Total amount")
    due_date: str = Field(description="Due date YYYY-MM-DD")


class OptionalFieldModel(BaseModel):
    """Model with optional fields."""

    name: str
    nickname: str | None = None
    age: int = 0


# ---------------------------------------------------------------------------
# OnFailAction tests
# ---------------------------------------------------------------------------


class TestOnFailAction:
    """Tests for the OnFailAction enum."""

    def test_values(self) -> None:
        assert OnFailAction.EXCEPTION.value == "exception"
        assert OnFailAction.REASK.value == "reask"
        assert OnFailAction.FILTER.value == "filter"
        assert OnFailAction.NOOP.value == "noop"

    def test_all_members(self) -> None:
        assert set(OnFailAction) == {
            OnFailAction.EXCEPTION,
            OnFailAction.REASK,
            OnFailAction.FILTER,
            OnFailAction.NOOP,
        }


# ---------------------------------------------------------------------------
# SchemaValidator tests
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    """Tests for SchemaValidator (Pydantic model validation)."""

    def test_valid_single_object(self) -> None:
        v = SchemaValidator(Invoice)
        output = json.dumps({
            "invoice_number": "INV-001",
            "amount": 100.0,
            "due_date": "2024-01-01",
        })
        result = v.validate(output)
        assert result.valid

    def test_valid_array_of_objects(self) -> None:
        v = SchemaValidator(Invoice)
        output = json.dumps([
            {
                "invoice_number": "INV-001",
                "amount": 100.0,
                "due_date": "2024-01-01",
            },
            {
                "invoice_number": "INV-002",
                "amount": 200.0,
                "due_date": "2024-02-01",
            },
        ])
        result = v.validate(output)
        assert result.valid

    def test_invalid_missing_required_field(self) -> None:
        v = SchemaValidator(Invoice)
        output = json.dumps({"invoice_number": "INV-001"})
        result = v.validate(output)
        assert not result.valid
        assert "amount" in (result.error_message or "")

    def test_invalid_wrong_type(self) -> None:
        v = SchemaValidator(Invoice, strict=True)
        output = json.dumps({
            "invoice_number": "INV-001",
            "amount": "not_a_number",
            "due_date": "2024-01-01",
        })
        result = v.validate(output)
        assert not result.valid

    def test_invalid_json(self) -> None:
        """Output that cannot be parsed as JSON should fail.

        Note: json-repair can fix many malformed inputs, so we
        test with something that repairs into a non-object scalar
        or still fails Pydantic validation.
        """
        v = SchemaValidator(Invoice)
        # Repairs into a plain string — no dict, no valid schema
        result = v.validate('"just a bare string"')
        assert not result.valid

    def test_non_strict_coerces_types(self) -> None:
        v = SchemaValidator(Invoice, strict=False)
        output = json.dumps({
            "invoice_number": "INV-001",
            "amount": "100.0",  # string that can be coerced to float
            "due_date": "2024-01-01",
        })
        result = v.validate(output)
        assert result.valid

    def test_rejects_non_basemodel(self) -> None:
        with pytest.raises(TypeError, match="Pydantic BaseModel"):
            SchemaValidator(dict)

    def test_schema_class_property(self) -> None:
        v = SchemaValidator(Invoice)
        assert v.schema_class is Invoice

    def test_on_fail_property(self) -> None:
        v = SchemaValidator(Invoice, on_fail=OnFailAction.FILTER)
        assert v.on_fail == OnFailAction.FILTER

    def test_default_on_fail_is_reask(self) -> None:
        v = SchemaValidator(Invoice)
        assert v.on_fail == OnFailAction.REASK

    def test_markdown_fenced_json(self) -> None:
        """Should handle markdown-fenced JSON output."""
        v = SchemaValidator(Invoice)
        output = (
            '```json\n'
            '{"invoice_number": "INV-001", '
            '"amount": 100.0, "due_date": "2024-01-01"}\n'
            '```'
        )
        result = v.validate(output)
        assert result.valid

    def test_multiple_items_one_invalid(self) -> None:
        """If one item in an array fails, the whole validation fails."""
        v = SchemaValidator(Invoice)
        output = json.dumps([
            {
                "invoice_number": "INV-001",
                "amount": 100.0,
                "due_date": "2024-01-01",
            },
            {"invoice_number": "INV-002"},  # missing amount, due_date
        ])
        result = v.validate(output)
        assert not result.valid
        assert "Item 1" in (result.error_message or "")

    def test_non_dict_item_in_array(self) -> None:
        v = SchemaValidator(Invoice)
        output = json.dumps(["not_an_object"])
        result = v.validate(output)
        assert not result.valid
        assert "expected object" in (result.error_message or "")


# ---------------------------------------------------------------------------
# ConfidenceThresholdValidator tests
# ---------------------------------------------------------------------------


class TestConfidenceThresholdValidator:
    """Tests for ConfidenceThresholdValidator."""

    def test_passes_above_threshold(self) -> None:
        v = ConfidenceThresholdValidator(min_confidence=0.5)
        output = json.dumps([
            {"name": "Alice", "confidence_score": 0.9},
            {"name": "Bob", "confidence_score": 0.8},
        ])
        result = v.validate(output)
        assert result.valid

    def test_fails_below_threshold(self) -> None:
        v = ConfidenceThresholdValidator(min_confidence=0.7)
        output = json.dumps([
            {"name": "Alice", "confidence_score": 0.9},
            {"name": "Bob", "confidence_score": 0.3},
        ])
        result = v.validate(output)
        assert not result.valid
        assert "0.3" in (result.error_message or "")

    def test_passes_without_score_key(self) -> None:
        """Items without the score key are not checked."""
        v = ConfidenceThresholdValidator(min_confidence=0.9)
        output = json.dumps({"name": "Alice"})
        result = v.validate(output)
        assert result.valid

    def test_custom_score_key(self) -> None:
        v = ConfidenceThresholdValidator(
            min_confidence=0.5, score_key="score"
        )
        output = json.dumps({"score": 0.3})
        result = v.validate(output)
        assert not result.valid

    def test_invalid_confidence_value(self) -> None:
        v = ConfidenceThresholdValidator(min_confidence=0.5)
        output = json.dumps({"confidence_score": "not_a_number"})
        result = v.validate(output)
        assert not result.valid
        assert "invalid confidence" in (result.error_message or "")

    def test_non_json_passes_through(self) -> None:
        """Non-JSON output passes (other validators handle JSON)."""
        v = ConfidenceThresholdValidator(min_confidence=0.5)
        result = v.validate("plain text")
        assert result.valid

    def test_invalid_min_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidenceThresholdValidator(min_confidence=1.5)

    def test_on_fail_default_is_filter(self) -> None:
        v = ConfidenceThresholdValidator(min_confidence=0.5)
        assert v.on_fail == OnFailAction.FILTER

    def test_single_object_below_threshold(self) -> None:
        v = ConfidenceThresholdValidator(min_confidence=0.8)
        output = json.dumps({"confidence_score": 0.5})
        result = v.validate(output)
        assert not result.valid


# ---------------------------------------------------------------------------
# FieldCompletenessValidator tests
# ---------------------------------------------------------------------------


class TestFieldCompletenessValidator:
    """Tests for FieldCompletenessValidator."""

    def test_passes_complete_object(self) -> None:
        v = FieldCompletenessValidator(Invoice)
        output = json.dumps({
            "invoice_number": "INV-001",
            "amount": 100.0,
            "due_date": "2024-01-01",
        })
        result = v.validate(output)
        assert result.valid

    def test_fails_missing_field(self) -> None:
        v = FieldCompletenessValidator(Invoice)
        output = json.dumps({"invoice_number": "INV-001"})
        result = v.validate(output)
        assert not result.valid
        assert "amount" in (result.error_message or "")
        assert "due_date" in (result.error_message or "")

    def test_fails_empty_string_field(self) -> None:
        v = FieldCompletenessValidator(Invoice)
        output = json.dumps({
            "invoice_number": "",
            "amount": 100.0,
            "due_date": "2024-01-01",
        })
        result = v.validate(output)
        assert not result.valid
        assert "empty" in (result.error_message or "").lower()

    def test_fails_none_field(self) -> None:
        v = FieldCompletenessValidator(Invoice)
        output = json.dumps({
            "invoice_number": None,
            "amount": 100.0,
            "due_date": "2024-01-01",
        })
        result = v.validate(output)
        assert not result.valid

    def test_optional_fields_ignored(self) -> None:
        """Fields with defaults or Optional type are not checked."""
        v = FieldCompletenessValidator(OptionalFieldModel)
        output = json.dumps({"name": "Alice"})
        result = v.validate(output)
        assert result.valid

    def test_rejects_non_basemodel(self) -> None:
        with pytest.raises(TypeError, match="Pydantic BaseModel"):
            FieldCompletenessValidator(dict)

    def test_invalid_json(self) -> None:
        """Non-object input should fail completeness check."""
        v = FieldCompletenessValidator(Invoice)
        # json-repair may fix syntax, but result won't be a valid
        # object with required fields.
        result = v.validate('"just a bare string"')
        assert not result.valid

    def test_array_of_objects(self) -> None:
        v = FieldCompletenessValidator(Invoice)
        output = json.dumps([
            {
                "invoice_number": "INV-001",
                "amount": 100.0,
                "due_date": "2024-01-01",
            },
            {"invoice_number": "INV-002"},  # missing amount, due_date
        ])
        result = v.validate(output)
        assert not result.valid
        assert "Item 1" in (result.error_message or "")

    def test_empty_list_field(self) -> None:
        """Empty list counts as empty for required fields."""

        class ListModel(BaseModel):
            items: list[str]

        v = FieldCompletenessValidator(ListModel)
        output = json.dumps({"items": []})
        result = v.validate(output)
        assert not result.valid
        assert "empty" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# ConsistencyValidator tests
# ---------------------------------------------------------------------------


class TestConsistencyValidator:
    """Tests for ConsistencyValidator."""

    @staticmethod
    def _dates_ordered(data: dict[str, Any]) -> str | None:
        start = data.get("start_date", "")
        end = data.get("end_date", "")
        if start and end and start > end:
            return "start_date must be before end_date"
        return None

    @staticmethod
    def _positive_amount(data: dict[str, Any]) -> str | None:
        amount = data.get("amount")
        if amount is not None and amount < 0:
            return "amount must be non-negative"
        return None

    def test_passes_consistent_data(self) -> None:
        v = ConsistencyValidator(rules=[self._dates_ordered])
        output = json.dumps({
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        })
        result = v.validate(output)
        assert result.valid

    def test_fails_inconsistent_data(self) -> None:
        v = ConsistencyValidator(rules=[self._dates_ordered])
        output = json.dumps({
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",
        })
        result = v.validate(output)
        assert not result.valid
        assert "start_date must be before end_date" in (
            result.error_message or ""
        )

    def test_multiple_rules(self) -> None:
        v = ConsistencyValidator(
            rules=[self._dates_ordered, self._positive_amount]
        )
        output = json.dumps({
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",
            "amount": -50,
        })
        result = v.validate(output)
        assert not result.valid
        assert "start_date" in (result.error_message or "")
        assert "amount" in (result.error_message or "")

    def test_empty_rules_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one rule"):
            ConsistencyValidator(rules=[])

    def test_non_json_input(self) -> None:
        """Non-object JSON input should fail gracefully."""
        v = ConsistencyValidator(rules=[self._dates_ordered])
        # json-repair may fix syntax; use a bare string that
        # becomes a non-dict after parsing.
        result = v.validate('"just a bare string"')
        # ConsistencyValidator skips non-dict items, so no rules
        # fire and it passes. This is expected behaviour.
        assert result.valid

    def test_array_of_objects(self) -> None:
        v = ConsistencyValidator(rules=[self._positive_amount])
        output = json.dumps([
            {"amount": 100},
            {"amount": -5},
        ])
        result = v.validate(output)
        assert not result.valid
        assert "Item 1" in (result.error_message or "")

    def test_on_fail_default_is_reask(self) -> None:
        v = ConsistencyValidator(rules=[self._dates_ordered])
        assert v.on_fail == OnFailAction.REASK


# ---------------------------------------------------------------------------
# ValidatorRegistry tests
# ---------------------------------------------------------------------------


class TestValidatorRegistry:
    """Tests for the validator registry pattern."""

    def test_register_and_retrieve(self) -> None:
        @register_validator(name="test_unique_validator_1")
        class _TestValidator(GuardrailValidator):
            def validate(self, output: str) -> ValidationResult:
                return ValidationResult(valid=True)

        cls = get_validator("test_unique_validator_1")
        assert cls is _TestValidator

    def test_duplicate_registration_raises(self) -> None:
        @register_validator(name="test_dup_check")
        class _First(GuardrailValidator):
            def validate(self, output: str) -> ValidationResult:
                return ValidationResult(valid=True)

        with pytest.raises(ValueError, match="already registered"):

            @register_validator(name="test_dup_check")
            class _Second(GuardrailValidator):
                def validate(self, output: str) -> ValidationResult:
                    return ValidationResult(valid=True)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="No validator registered"):
            get_validator("nonexistent_validator_xyz")

    def test_list_validators_includes_registered(self) -> None:
        @register_validator(name="test_list_check")
        class _Listed(GuardrailValidator):
            def validate(self, output: str) -> ValidationResult:
                return ValidationResult(valid=True)

        all_validators = list_validators()
        assert "test_list_check" in all_validators

    def test_register_non_validator_raises(self) -> None:
        with pytest.raises(TypeError, match="GuardrailValidator"):
            @register_validator(name="test_bad_type")
            class _NotAValidator:
                pass


# ---------------------------------------------------------------------------
# ValidatorChain tests
# ---------------------------------------------------------------------------


class _PassValidator(GuardrailValidator):
    def validate(self, output: str) -> ValidationResult:
        return ValidationResult(valid=True)


class _FailValidator(GuardrailValidator):
    def __init__(self, msg: str = "fail") -> None:
        self._msg = msg

    def validate(self, output: str) -> ValidationResult:
        return ValidationResult(valid=False, error_message=self._msg)


class TestValidatorChain:
    """Tests for ValidatorChain composition."""

    def test_all_pass(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(_PassValidator(), OnFailAction.REASK),
            ValidatorEntry(_PassValidator(), OnFailAction.FILTER),
        ])
        result = chain.run("output")
        assert result.passed
        assert not result.should_reask
        assert not result.should_filter

    def test_reask_failure(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(_FailValidator("bad"), OnFailAction.REASK),
        ])
        result = chain.run("output")
        assert not result.passed
        assert result.should_reask
        assert not result.should_filter

    def test_filter_failure(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(_FailValidator("low"), OnFailAction.FILTER),
        ])
        result = chain.run("output")
        assert not result.passed
        assert result.should_filter

    def test_exception_failure_raises(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(
                _FailValidator("critical"),
                OnFailAction.EXCEPTION,
            ),
        ])
        with pytest.raises(GuardrailValidationError, match="critical"):
            chain.run("output")

    def test_noop_records_but_passes(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(_FailValidator("minor"), OnFailAction.NOOP),
        ])
        result = chain.run("output")
        assert not result.passed  # failure recorded
        assert not result.should_reask
        assert not result.should_filter
        assert len(result.failures) == 1

    def test_mixed_actions(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(_PassValidator(), OnFailAction.REASK),
            ValidatorEntry(_FailValidator("a"), OnFailAction.NOOP),
            ValidatorEntry(_FailValidator("b"), OnFailAction.FILTER),
        ])
        result = chain.run("output")
        assert not result.passed
        assert result.should_filter
        assert not result.should_reask
        assert len(result.failures) == 2

    def test_error_messages(self) -> None:
        chain = ValidatorChain([
            ValidatorEntry(_FailValidator("err1"), OnFailAction.NOOP),
            ValidatorEntry(_FailValidator("err2"), OnFailAction.NOOP),
        ])
        result = chain.run("output")
        assert result.error_messages == ["err1", "err2"]

    def test_tuple_entries(self) -> None:
        """Accepts (validator, action) tuples."""
        chain = ValidatorChain([
            (_PassValidator(), OnFailAction.REASK),
        ])
        assert len(chain.entries) == 1

    def test_as_guardrail_validators(self) -> None:
        v1, v2 = _PassValidator(), _FailValidator()
        chain = ValidatorChain([
            ValidatorEntry(v1, OnFailAction.REASK),
            ValidatorEntry(v2, OnFailAction.FILTER),
        ])
        validators = chain.as_guardrail_validators()
        assert validators == [v1, v2]

    def test_invalid_entry_type_raises(self) -> None:
        with pytest.raises(TypeError):
            ValidatorChain(["not_a_valid_entry"])


# ---------------------------------------------------------------------------
# ValidatorEntry tests
# ---------------------------------------------------------------------------


class TestValidatorEntry:
    """Tests for ValidatorEntry."""

    def test_repr(self) -> None:
        entry = ValidatorEntry(
            _PassValidator(), OnFailAction.REASK
        )
        assert "_PassValidator" in repr(entry)
        assert "reask" in repr(entry)

    def test_default_on_fail(self) -> None:
        entry = ValidatorEntry(_PassValidator())
        assert entry.on_fail == OnFailAction.EXCEPTION


# ---------------------------------------------------------------------------
# ChainResult tests
# ---------------------------------------------------------------------------


class TestChainResult:
    """Tests for ChainResult."""

    def test_passed_result(self) -> None:
        result = ChainResult(
            passed=True,
            failures=[],
            should_reask=False,
            should_filter=False,
        )
        assert result.passed
        assert result.error_messages == []

    def test_failed_result_with_messages(self) -> None:
        entry = ValidatorEntry(_FailValidator("x"), OnFailAction.NOOP)
        vr = ValidationResult(valid=False, error_message="x")
        result = ChainResult(
            passed=False,
            failures=[(entry, vr)],
            should_reask=False,
            should_filter=False,
        )
        assert result.error_messages == ["x"]


# ---------------------------------------------------------------------------
# GuardrailValidationError tests
# ---------------------------------------------------------------------------


class TestGuardrailValidationError:
    """Tests for GuardrailValidationError exception."""

    def test_attributes(self) -> None:
        v = _FailValidator("test")
        vr = ValidationResult(valid=False, error_message="test")
        exc = GuardrailValidationError("msg", validator=v, result=vr)
        assert exc.validator is v
        assert exc.result is vr
        assert str(exc) == "msg"


# ---------------------------------------------------------------------------
# Integration: GuardrailLanguageModel with new validators
# ---------------------------------------------------------------------------


class TestGuardrailWithNewValidators:
    """Integration tests for GuardrailLanguageModel + new validators."""

    def test_schema_validator_retries_on_failure(self) -> None:
        """SchemaValidator should cause retries on validation failure."""
        from langcore.core.base_model import BaseLanguageModel
        from langcore.core.types import ScoredOutput

        class _Inner(BaseLanguageModel):
            def __init__(self):
                super().__init__()
                self._call = 0

            def infer(self, batch_prompts, **kw):
                for _ in batch_prompts:
                    self._call += 1
                    if self._call == 1:
                        yield [ScoredOutput(
                            score=1.0,
                            output='{"invoice_number": "INV-001"}',
                        )]
                    else:
                        yield [ScoredOutput(
                            score=1.0,
                            output=json.dumps({
                                "invoice_number": "INV-001",
                                "amount": 100.0,
                                "due_date": "2024-01-01",
                            }),
                        )]

            async def async_infer(self, batch_prompts, **kw):
                return []

        from langcore_guardrails import GuardrailLanguageModel

        inner = _Inner()
        guard = GuardrailLanguageModel(
            model_id="guardrails/test",
            inner=inner,
            validators=[SchemaValidator(Invoice)],
            max_retries=3,
        )
        results = list(guard.infer(["extract invoice"]))
        assert results[0][0].score == 1.0
        # First attempt fails validation, second succeeds
        assert inner._call == 2

    def test_confidence_threshold_filters_low_confidence(self) -> None:
        """Demo of confidence threshold + JSON schema working together."""
        from langcore_guardrails import JsonSchemaValidator

        v_json = JsonSchemaValidator()
        v_conf = ConfidenceThresholdValidator(min_confidence=0.8)

        # High confidence — both pass
        high = json.dumps({"name": "Alice", "confidence_score": 0.95})
        assert v_json.validate(high).valid
        assert v_conf.validate(high).valid

        # Low confidence — confidence check fails
        low = json.dumps({"name": "Bob", "confidence_score": 0.3})
        assert v_json.validate(low).valid
        assert not v_conf.validate(low).valid
