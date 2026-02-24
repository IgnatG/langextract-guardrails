"""Tests for GroundingValidator."""

# ruff: noqa: RUF059 RUF012

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import pytest

from langcore_guardrails import GroundingValidator, OnFailAction, ValidationResult

# ---------------------------------------------------------------------------
# Minimal stubs — avoid importing langcore.core.data so tests stay
# lightweight and isolated to the guardrails package.
# ---------------------------------------------------------------------------


class _AlignmentStatus(enum.Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


@dataclasses.dataclass
class _CharInterval:
    start_pos: int | None = None
    end_pos: int | None = None


@dataclasses.dataclass
class _Extraction:
    extraction_class: str = "entity"
    extraction_text: str = ""
    char_interval: _CharInterval | None = None
    alignment_status: _AlignmentStatus | None = None
    confidence_score: float | None = None
    attributes: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for __init__ validation."""

    def test_default_params(self):
        v = GroundingValidator()
        assert v.min_alignment_quality == "MATCH_FUZZY"
        assert v.min_coverage == 0.5
        assert v.on_fail == OnFailAction.FILTER

    def test_custom_params(self):
        v = GroundingValidator(
            min_alignment_quality="MATCH_EXACT",
            min_coverage=0.8,
            on_fail=OnFailAction.EXCEPTION,
        )
        assert v.min_alignment_quality == "MATCH_EXACT"
        assert v.min_coverage == 0.8
        assert v.on_fail == OnFailAction.EXCEPTION

    def test_case_insensitive_quality(self):
        v = GroundingValidator(min_alignment_quality="match_lesser")
        assert v.min_alignment_quality == "MATCH_LESSER"

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="min_alignment_quality"):
            GroundingValidator(min_alignment_quality="UNKNOWN")

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError, match="min_coverage"):
            GroundingValidator(min_coverage=1.5)

    def test_negative_coverage_raises(self):
        with pytest.raises(ValueError, match="min_coverage"):
            GroundingValidator(min_coverage=-0.1)


# ---------------------------------------------------------------------------
# Raw-output mode (no-op)
# ---------------------------------------------------------------------------


class TestValidateRawOutput:
    """The validate() method always passes (grounding is post-alignment)."""

    def test_always_passes(self):
        v = GroundingValidator()
        result = v.validate("any raw output")
        assert result == ValidationResult(valid=True)

    def test_empty_string_passes(self):
        v = GroundingValidator()
        assert v.validate("").valid is True


# ---------------------------------------------------------------------------
# Post-alignment validation — alignment quality checks
# ---------------------------------------------------------------------------


class TestAlignmentQualityFiltering:
    """Tests for alignment quality threshold."""

    def test_exact_match_passes_all_thresholds(self):
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 9),
        )
        for quality in ("MATCH_EXACT", "MATCH_GREATER", "MATCH_LESSER", "MATCH_FUZZY"):
            v = GroundingValidator(min_alignment_quality=quality, min_coverage=0.0)
            passed, filtered = v.validate_extractions([ext], source_text="Acme Corp")
            assert len(passed) == 1, f"Should pass for threshold {quality}"

    def test_fuzzy_match_rejected_by_exact_threshold(self):
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_FUZZY,
            char_interval=_CharInterval(0, 9),
        )
        v = GroundingValidator(min_alignment_quality="MATCH_EXACT", min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext], source_text="Acme Corp")
        assert len(passed) == 0
        assert len(filtered) == 1

    def test_fuzzy_match_rejected_by_lesser_threshold(self):
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_FUZZY,
            char_interval=_CharInterval(0, 9),
        )
        v = GroundingValidator(min_alignment_quality="MATCH_LESSER", min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext])
        assert len(filtered) == 1

    def test_lesser_match_passes_lesser_threshold(self):
        ext = _Extraction(
            extraction_text="Acme",
            alignment_status=_AlignmentStatus.MATCH_LESSER,
            char_interval=_CharInterval(0, 4),
        )
        v = GroundingValidator(min_alignment_quality="MATCH_LESSER", min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext])
        assert len(passed) == 1

    def test_greater_match_passes_greater_threshold(self):
        ext = _Extraction(
            extraction_text="Acme Corp Inc",
            alignment_status=_AlignmentStatus.MATCH_GREATER,
            char_interval=_CharInterval(0, 13),
        )
        v = GroundingValidator(min_alignment_quality="MATCH_GREATER", min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext])
        assert len(passed) == 1


# ---------------------------------------------------------------------------
# Post-alignment validation — unaligned extractions
# ---------------------------------------------------------------------------


class TestUnalignedExtractions:
    """Tests for extractions with no alignment status."""

    def test_no_alignment_status_rejected(self):
        ext = _Extraction(extraction_text="hallucinated", alignment_status=None)
        v = GroundingValidator(min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext])
        assert len(filtered) == 1
        assert len(passed) == 0

    def test_mixed_aligned_and_unaligned(self):
        good = _Extraction(
            extraction_text="Acme",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 4),
        )
        bad = _Extraction(extraction_text="hallucinated", alignment_status=None)
        v = GroundingValidator(min_coverage=0.0)
        passed, filtered = v.validate_extractions([good, bad])
        assert len(passed) == 1
        assert passed[0].extraction_text == "Acme"
        assert len(filtered) == 1
        assert filtered[0].extraction_text == "hallucinated"


# ---------------------------------------------------------------------------
# Post-alignment validation — coverage checks
# ---------------------------------------------------------------------------


class TestCoverageFiltering:
    """Tests for character coverage threshold."""

    def test_full_coverage_passes(self):
        text = "Acme Corp"
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 9),
        )
        v = GroundingValidator(min_coverage=1.0)
        passed, filtered = v.validate_extractions([ext], source_text=text)
        assert len(passed) == 1

    def test_low_coverage_rejected(self):
        text = "The Acme Corporation signed the deal."
        ext = _Extraction(
            extraction_text="Acme Corporation",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            # Span only covers 4 chars out of 16-char extraction text
            char_interval=_CharInterval(4, 8),
        )
        v = GroundingValidator(min_coverage=0.5)
        passed, filtered = v.validate_extractions([ext], source_text=text)
        assert len(filtered) == 1

    def test_partial_coverage_above_threshold_passes(self):
        text = "Acme Corp is great."
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            # Span covers 7 chars of 9-char extraction = 0.78
            char_interval=_CharInterval(0, 7),
        )
        v = GroundingValidator(min_coverage=0.5)
        passed, filtered = v.validate_extractions([ext], source_text=text)
        assert len(passed) == 1

    def test_no_source_text_skips_coverage_check(self):
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 3),  # Would fail coverage
        )
        v = GroundingValidator(min_coverage=0.9)
        passed, filtered = v.validate_extractions([ext], source_text=None)
        assert len(passed) == 1

    def test_no_char_interval_skips_coverage_check(self):
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=None,
        )
        v = GroundingValidator(min_coverage=0.9)
        passed, filtered = v.validate_extractions([ext], source_text="some text")
        assert len(passed) == 1

    def test_zero_coverage_threshold_skips_check(self):
        ext = _Extraction(
            extraction_text="Acme Corp",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 1),  # Very low coverage
        )
        v = GroundingValidator(min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext], source_text="Acme Corp")
        assert len(passed) == 1


# ---------------------------------------------------------------------------
# validate_document convenience method
# ---------------------------------------------------------------------------


class TestValidateDocument:
    """Tests for the validate_document convenience method."""

    def test_reads_extractions_and_text(self):
        ext = _Extraction(
            extraction_text="Acme",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 4),
        )

        class _FakeDoc:
            extractions = [ext]
            text = "Acme Corp"

        v = GroundingValidator(min_coverage=0.0)
        passed, filtered = v.validate_document(_FakeDoc())
        assert len(passed) == 1

    def test_none_extractions_returns_empty(self):
        class _FakeDoc:
            extractions = None
            text = "something"

        v = GroundingValidator()
        passed, filtered = v.validate_document(_FakeDoc())
        assert passed == []
        assert filtered == []


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for validate_extractions."""

    def test_empty_list_returns_empty(self):
        v = GroundingValidator()
        passed, filtered = v.validate_extractions([])
        assert passed == []
        assert filtered == []

    def test_empty_extraction_text_passes_coverage(self):
        """Empty extraction text means coverage can't be computed — pass."""
        ext = _Extraction(
            extraction_text="",
            alignment_status=_AlignmentStatus.MATCH_EXACT,
            char_interval=_CharInterval(0, 0),
        )
        v = GroundingValidator(min_coverage=0.5)
        passed, filtered = v.validate_extractions([ext], source_text="text")
        assert len(passed) == 1

    def test_all_extractions_filtered(self):
        exts = [
            _Extraction(extraction_text="a", alignment_status=None),
            _Extraction(extraction_text="b", alignment_status=None),
        ]
        v = GroundingValidator()
        passed, filtered = v.validate_extractions(exts)
        assert len(passed) == 0
        assert len(filtered) == 2

    def test_all_extractions_pass(self):
        exts = [
            _Extraction(
                extraction_text="a",
                alignment_status=_AlignmentStatus.MATCH_EXACT,
                char_interval=_CharInterval(0, 1),
            ),
            _Extraction(
                extraction_text="b",
                alignment_status=_AlignmentStatus.MATCH_EXACT,
                char_interval=_CharInterval(2, 3),
            ),
        ]
        v = GroundingValidator(min_coverage=0.0)
        passed, filtered = v.validate_extractions(exts, source_text="a b")
        assert len(passed) == 2
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Integration with langcore.core.data types (if available)
# ---------------------------------------------------------------------------


class TestWithLangcoreTypes:
    """Tests using actual langcore data types."""

    def test_with_real_alignment_status(self):
        """Use the real AlignmentStatus enum from langcore."""
        from langcore.core.data import AlignmentStatus, CharInterval, Extraction

        ext = Extraction(
            extraction_class="entity",
            extraction_text="Acme Corp",
            alignment_status=AlignmentStatus.MATCH_EXACT,
            char_interval=CharInterval(start_pos=0, end_pos=9),
        )
        v = GroundingValidator(min_coverage=0.5)
        passed, filtered = v.validate_extractions(
            [ext], source_text="Acme Corp is here"
        )
        assert len(passed) == 1

    def test_real_fuzzy_rejected_by_exact_threshold(self):
        from langcore.core.data import AlignmentStatus, CharInterval, Extraction

        ext = Extraction(
            extraction_class="entity",
            extraction_text="Acme Corp",
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            char_interval=CharInterval(start_pos=0, end_pos=9),
        )
        v = GroundingValidator(min_alignment_quality="MATCH_EXACT", min_coverage=0.0)
        passed, filtered = v.validate_extractions([ext])
        assert len(filtered) == 1

    def test_real_unaligned_extraction(self):
        from langcore.core.data import Extraction

        ext = Extraction(
            extraction_class="entity",
            extraction_text="hallucinated",
            alignment_status=None,
        )
        v = GroundingValidator()
        passed, filtered = v.validate_extractions([ext])
        assert len(filtered) == 1
        assert len(passed) == 0
