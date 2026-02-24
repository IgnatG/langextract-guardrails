"""Guardrail provider implementation.

Wraps any ``BaseLanguageModel`` with output validation and retry
logic.  On validation failure, a corrective prompt is constructed
that includes the original prompt, the invalid output, and the
validation error.  The inner provider is re-invoked up to
``max_retries`` times.

Supports per-validator ``OnFailAction`` semantics via
``ValidatorChain``:

* **REASK** — re-prompt the LLM with validation error feedback.
* **FILTER** — silently discard the output (return score 0).
* **EXCEPTION** — raise immediately.
* **NOOP** — log and continue without re-prompting.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import langcore as lx
from langcore.core.base_model import BaseLanguageModel
from langcore.core.types import ScoredOutput

from langcore_guardrails.validators import (
    GuardrailValidator,
    OnFailAction,
)

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

# Template for error-only correction mode: omits the invalid
# output entirely to reduce token usage and avoid the model
# "fixating" on junk output.
_ERROR_ONLY_CORRECTION_TEMPLATE = (
    "Your previous response was invalid.\n\n"
    "--- Original prompt ---\n"
    "{original_prompt}\n\n"
    "--- Validation error ---\n"
    "{error_message}\n\n"
    "Please try again. "
    "Return ONLY the corrected output, nothing else."
)


from dataclasses import dataclass
from dataclasses import field as dc_field


@dataclass(frozen=True, slots=True)
class _ValidationOutcome:
    """Internal result of running all validators on an output.

    Aggregates per-validator ``OnFailAction`` outcomes so that the
    retry loop can decide what to do.
    """

    all_passed: bool
    reask_errors: list[str] = dc_field(default_factory=list)
    should_filter: bool = False

    @property
    def combined_error_message(self) -> str:
        """Join all reask error messages into a single string."""
        return "\n".join(self.reask_errors)


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
        max_concurrency: Maximum number of prompts to process
            concurrently in ``async_infer``.  ``None`` (default)
            means unlimited.
        max_correction_prompt_length: Truncate the original prompt
            in correction prompts to this many characters.
            ``None`` (default) means no truncation.
        max_correction_output_length: Truncate the invalid output
            in correction prompts to this many characters.
            ``None`` (default) means no truncation.
        include_output_in_correction: When ``True`` (default),
            the invalid output is included in the correction
            prompt.  Set to ``False`` for "error-only" mode,
            which omits the invalid output entirely to reduce
            token usage and avoid the model fixating on bad
            output.
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
        max_concurrency: int | None = None,
        max_correction_prompt_length: int | None = None,
        max_correction_output_length: int | None = None,
        include_output_in_correction: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self._inner = inner
        self._validators = validators
        self._max_retries = max_retries
        self._include_output_in_correction = include_output_in_correction
        # Use error-only template when output is excluded, unless
        # the caller supplied a fully custom template.
        if correction_template is not None:
            self._correction_template = correction_template
        elif include_output_in_correction:
            self._correction_template = _CORRECTION_TEMPLATE
        else:
            self._correction_template = _ERROR_ONLY_CORRECTION_TEMPLATE
        self._max_concurrency = max_concurrency
        self._max_correction_prompt_length = max_correction_prompt_length
        self._max_correction_output_length = max_correction_output_length

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

    @staticmethod
    def _truncate(text: str, max_length: int | None) -> str:
        """Truncate text to max_length if set.

        Parameters:
            text: The text to truncate.
            max_length: Maximum chars, or ``None`` for no limit.

        Returns:
            The (possibly truncated) string.
        """
        if max_length is None or len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def _pick_best(
        outputs: list[ScoredOutput],
    ) -> ScoredOutput:
        """Select the highest-scored output from a list.

        Parameters:
            outputs: Non-empty list of scored outputs.

        Returns:
            The ``ScoredOutput`` with the highest score.
        """
        return max(
            outputs,
            key=lambda o: o.score if o.score is not None else 0.0,
        )

    def _validate(self, output: str) -> _ValidationOutcome:
        """Run all validators against the output, respecting ``OnFailAction``.

        Each validator is checked for an ``on_fail`` attribute.  If
        absent, the default action is ``REASK`` (re-prompt).

        Returns a ``_ValidationOutcome`` summarising:
        - **reask_errors** — error messages requiring a retry.
        - **should_filter** — output should be silently discarded.
        - **all_passed** — every validator accepted the output.

        Parameters:
            output: The raw output string to validate.

        Returns:
            A ``_ValidationOutcome`` with per-action aggregation.

        Raises:
            GuardrailValidationError: If any validator with
                ``OnFailAction.EXCEPTION`` fails.
        """
        from langcore_guardrails.validator_registry import (
            GuardrailValidationError,
        )

        reask_errors: list[str] = []
        should_filter = False

        for validator in self._validators:
            result = validator.validate(output)
            if result.valid:
                continue

            action = getattr(validator, "on_fail", OnFailAction.REASK)
            if isinstance(action, str):
                action = OnFailAction(action)

            if action == OnFailAction.EXCEPTION:
                raise GuardrailValidationError(
                    f"{validator.__class__.__name__} failed: "
                    f"{result.error_message}",
                    validator=validator,
                    result=result,
                )
            elif action == OnFailAction.FILTER:
                should_filter = True
                logger.info(
                    "Validator %s failed (action=FILTER): %s",
                    validator.__class__.__name__,
                    result.error_message,
                )
            elif action == OnFailAction.NOOP:
                logger.info(
                    "Validator %s failed (action=NOOP): %s",
                    validator.__class__.__name__,
                    result.error_message,
                )
            else:
                # REASK (default)
                if result.error_message:
                    reask_errors.append(result.error_message)

        return _ValidationOutcome(
            all_passed=not reask_errors and not should_filter,
            reask_errors=reask_errors,
            should_filter=should_filter,
        )

    def _build_correction_prompt(
        self,
        original_prompt: str,
        invalid_output: str,
        error_message: str,
    ) -> str:
        """Build a corrective prompt for retry.

        Applies truncation to the original prompt and invalid
        output when ``max_correction_prompt_length`` /
        ``max_correction_output_length`` are set.  When
        ``include_output_in_correction`` is ``False``, the
        invalid output is omitted entirely.

        Parameters:
            original_prompt: The original user prompt.
            invalid_output: The invalid output from the LLM.
            error_message: The validation error description.

        Returns:
            A corrective prompt string.
        """
        fmt_kwargs: dict[str, str] = {
            "original_prompt": self._truncate(
                original_prompt,
                self._max_correction_prompt_length,
            ),
            "error_message": error_message,
        }
        if self._include_output_in_correction:
            fmt_kwargs["invalid_output"] = self._truncate(
                invalid_output,
                self._max_correction_output_length,
            )
        return self._correction_template.format(**fmt_kwargs)

    def _infer_single_with_retries(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> list[ScoredOutput]:
        """Run inference for a single prompt with retry logic.

        Respects per-validator ``OnFailAction``:
        - **REASK**: re-prompt with error feedback.
        - **FILTER**: return output with ``score=0.0``.
        - **EXCEPTION**: raised immediately by ``_validate``.
        - **NOOP**: logged only; does not trigger retry.

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
            first_result = next(self._inner.infer([current_prompt], **kwargs), None)
            if not first_result:
                logger.warning(
                    "Attempt %d/%d: inner provider returned " "no results for prompt",
                    attempt + 1,
                    1 + self._max_retries,
                )
                continue

            outputs = list(first_result)
            best = self._pick_best(outputs)
            output_text = best.output or ""

            # Validate (may raise for EXCEPTION action)
            outcome = self._validate(output_text)
            if outcome.all_passed:
                others = [o for o in outputs if o is not best]
                return [best, *others]

            # FILTER action — discard immediately, no retry
            if outcome.should_filter and not outcome.reask_errors:
                logger.info(
                    "Attempt %d/%d: output filtered by validator",
                    attempt + 1,
                    1 + self._max_retries,
                )
                return [ScoredOutput(score=0.0, output=output_text, usage=best.usage)]

            # REASK action — build corrective prompt
            logger.info(
                "Attempt %d/%d: validation failed — %s",
                attempt + 1,
                1 + self._max_retries,
                outcome.combined_error_message,
            )
            if attempt < self._max_retries and outcome.reask_errors:
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    invalid_output=output_text,
                    error_message=outcome.combined_error_message,
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

        Respects per-validator ``OnFailAction`` — see
        :meth:`_infer_single_with_retries` for details.

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
            best = self._pick_best(outputs)
            output_text = best.output or ""

            # Validate (may raise for EXCEPTION action)
            outcome = self._validate(output_text)
            if outcome.all_passed:
                others = [o for o in outputs if o is not best]
                return [best, *others]

            # FILTER action — discard immediately, no retry
            if outcome.should_filter and not outcome.reask_errors:
                logger.info(
                    "Attempt %d/%d: output filtered by validator (async)",
                    attempt + 1,
                    1 + self._max_retries,
                )
                return [ScoredOutput(score=0.0, output=output_text, usage=best.usage)]

            # REASK action — build corrective prompt
            logger.info(
                "Attempt %d/%d: validation failed (async) — %s",
                attempt + 1,
                1 + self._max_retries,
                outcome.combined_error_message,
            )
            if attempt < self._max_retries and outcome.reask_errors:
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    invalid_output=output_text,
                    error_message=outcome.combined_error_message,
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
        if self._max_concurrency is not None:
            sem = asyncio.Semaphore(self._max_concurrency)

            async def _limited(
                coro: Any,
            ) -> list[ScoredOutput]:
                async with sem:
                    return await coro

            tasks = [_limited(t) for t in tasks]
        return list(await asyncio.gather(*tasks))
