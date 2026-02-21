"""Guardrail provider implementation.

Wraps any ``BaseLanguageModel`` with output validation and retry
logic.  On validation failure, a corrective prompt is constructed
that includes the original prompt, the invalid output, and the
validation error.  The inner provider is re-invoked up to
``max_retries`` times.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import langextract as lx
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from langextract_guardrails.validators import GuardrailValidator, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

_CORRECTION_TEMPLATE = (
    "Your previous response was invalid.\n\n"
    "--- Original prompt ---\n"
    "{original_prompt}\n\n"
    "--- Your invalid response ---\n"
    "{invalid_output}\n\n"
    "--- Validation error ---\n"
    "{error_message}\n\n"
    "Please correct your response and try again. "
    "Return ONLY the corrected output, nothing else."
)


@lx.providers.registry.register(r"^guardrails", priority=5)
class GuardrailLanguageModel(BaseLanguageModel):
    """Provider that validates LLM output and retries on failure.

    Wraps any ``BaseLanguageModel`` and applies a chain of
    ``GuardrailValidator`` instances to each response.  When
    validation fails, the provider constructs a corrective prompt
    and re-invokes the inner provider, up to ``max_retries`` times.

    This subsumes the retry-optimisation (#14) and verification
    (#18) provider concepts from the provider ideas assessment.

    Parameters:
        model_id: The model identifier (typically prefixed with
            ``guardrails/``).
        inner: The ``BaseLanguageModel`` instance to wrap.
        validators: A list of ``GuardrailValidator`` instances to
            apply in order.
        max_retries: Maximum number of retry attempts on validation
            failure. Defaults to ``3``.
        correction_template: Template for the corrective prompt.
            Must contain ``{original_prompt}``,
            ``{invalid_output}``, and ``{error_message}``
            placeholders.
        **kwargs: Additional keyword arguments forwarded to the
            base class.
    """

    def __init__(
        self,
        model_id: str,
        *,
        inner: BaseLanguageModel,
        validators: list[GuardrailValidator],
        max_retries: int = 3,
        correction_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self._inner = inner
        self._validators = validators
        self._max_retries = max_retries
        self._correction_template = correction_template or _CORRECTION_TEMPLATE

    # -- Public accessors --

    @property
    def inner(self) -> BaseLanguageModel:
        """Return the wrapped inner provider.

        Returns:
            The inner ``BaseLanguageModel`` instance.
        """
        return self._inner

    @property
    def validators(self) -> list[GuardrailValidator]:
        """Return a copy of the configured validators.

        Returns:
            A list of ``GuardrailValidator`` instances.
        """
        return list(self._validators)

    @property
    def max_retries(self) -> int:
        """Return the maximum number of retries.

        Returns:
            The configured max retry count.
        """
        return self._max_retries

    # -- Private helpers --

    def _validate(self, output: str) -> ValidationResult:
        """Run all validators against the output.

        Returns the first failing result, or a passing result if
        all validators pass.

        Parameters:
            output: The raw output string to validate.

        Returns:
            A ``ValidationResult`` from the first failing validator,
            or a passing result.
        """
        for validator in self._validators:
            result = validator.validate(output)
            if not result.valid:
                return result
        return ValidationResult(valid=True)

    def _build_correction_prompt(
        self,
        original_prompt: str,
        invalid_output: str,
        error_message: str,
    ) -> str:
        """Build a corrective prompt for retry.

        Parameters:
            original_prompt: The original user prompt.
            invalid_output: The invalid output from the LLM.
            error_message: The validation error description.

        Returns:
            A corrective prompt string.
        """
        return self._correction_template.format(
            original_prompt=original_prompt,
            invalid_output=invalid_output,
            error_message=error_message,
        )

    def _infer_single_with_retries(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> list[ScoredOutput]:
        """Run inference for a single prompt with retry logic.

        Parameters:
            prompt: The prompt string.
            **kwargs: Additional kwargs forwarded to the inner
                provider.

        Returns:
            A list of ``ScoredOutput`` for this prompt.
        """
        current_prompt = prompt

        for attempt in range(1 + self._max_retries):
            # Get the result from the inner provider (single prompt)
            results = list(self._inner.infer([current_prompt], **kwargs))
            if not results or not results[0]:
                logger.warning(
                    "Attempt %d/%d: inner provider returned no results for prompt",
                    attempt + 1,
                    1 + self._max_retries,
                )
                continue

            outputs = list(results[0])
            best = outputs[0]
            output_text = best.output or ""

            # Validate
            validation = self._validate(output_text)
            if validation.valid:
                return outputs

            # Log the failure
            logger.info(
                "Attempt %d/%d: validation failed — %s",
                attempt + 1,
                1 + self._max_retries,
                validation.error_message,
            )

            # Build corrective prompt for next attempt
            if attempt < self._max_retries:
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    invalid_output=output_text,
                    error_message=validation.error_message or "",
                )

        # All retries exhausted — return last result with score 0
        logger.warning(
            "All %d attempts failed validation for prompt",
            1 + self._max_retries,
        )
        return [
            ScoredOutput(
                score=0.0,
                output=output_text,
                usage=best.usage,
            )
        ]

    async def _async_infer_single_with_retries(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> list[ScoredOutput]:
        """Async inference for a single prompt with retry logic.

        Parameters:
            prompt: The prompt string.
            **kwargs: Additional kwargs forwarded to the inner
                provider.

        Returns:
            A list of ``ScoredOutput`` for this prompt.
        """
        current_prompt = prompt
        output_text = ""
        best = ScoredOutput(score=0.0, output="")

        for attempt in range(1 + self._max_retries):
            results = await self._inner.async_infer([current_prompt], **kwargs)
            if not results or not results[0]:
                logger.warning(
                    "Attempt %d/%d: inner provider returned no "
                    "results for prompt (async)",
                    attempt + 1,
                    1 + self._max_retries,
                )
                continue

            outputs = list(results[0])
            best = outputs[0]
            output_text = best.output or ""

            validation = self._validate(output_text)
            if validation.valid:
                return outputs

            logger.info(
                "Attempt %d/%d: validation failed (async) — %s",
                attempt + 1,
                1 + self._max_retries,
                validation.error_message,
            )

            if attempt < self._max_retries:
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    invalid_output=output_text,
                    error_message=validation.error_message or "",
                )

        logger.warning(
            "All %d attempts failed validation for prompt (async)",
            1 + self._max_retries,
        )
        return [
            ScoredOutput(
                score=0.0,
                output=output_text,
                usage=best.usage,
            )
        ]

    # -- BaseLanguageModel interface --

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Inference with validation and retry for each prompt.

        Each prompt in the batch is processed independently.  If
        validation fails, a corrective prompt is sent up to
        ``max_retries`` times before yielding the final result
        with ``score=0.0``.

        Parameters:
            batch_prompts: A sequence of prompt strings.
            **kwargs: Additional keyword arguments forwarded to the
                inner provider.

        Yields:
            Sequences of ``ScoredOutput`` per prompt.
        """
        for prompt in batch_prompts:
            yield self._infer_single_with_retries(prompt, **kwargs)

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> list[Sequence[ScoredOutput]]:
        """Async inference with validation and retry for each prompt.

        All prompts are processed concurrently.  Each prompt may
        individually trigger up to ``max_retries`` corrective
        retries.

        Parameters:
            batch_prompts: A sequence of prompt strings.
            **kwargs: Additional keyword arguments forwarded to the
                inner provider.

        Returns:
            A list of ``ScoredOutput`` sequences per prompt.
        """
        tasks = [
            self._async_infer_single_with_retries(prompt, **kwargs)
            for prompt in batch_prompts
        ]
        return list(await asyncio.gather(*tasks))
