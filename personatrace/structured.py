"""Pydantic-validated LLM output with one more attempt.

The useful part is not "retry three times"; it is that the retry receives the
validation error. A model that returned ``importance: 12`` or a self-loop edge
gets a concrete correction instead of silently poisoning the graph.

Usage::

    result = call_structured(llm, prompt, GraphEdgeBatch, retries=2)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, last_error: Any = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def call_structured(
    llm: Any,
    prompt: str,
    model_cls: Type[T],
    *,
    retries: int = 2,
    max_tokens: int = 1200,
) -> T:
    """Ask for JSON, validate with Pydantic, and retry with the error text.

    ``retries`` is the number of *additional* attempts (2 means at most 3
    calls). Network/JSON errors are retried too; the conversation continues
    only after a valid object or the final failure, which the caller may catch.
    """
    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    messages = [{
        "role": "user",
        "content": (
            f"{prompt}\n\n"
            "Return ONLY one JSON object. It must match this JSON Schema:\n"
            f"{schema}"
        ),
    }]
    last_error: Any = None
    attempts = max(0, int(retries)) + 1

    for attempt in range(1, attempts + 1):
        payload: Optional[dict] = None
        try:
            payload = llm.chat_json(messages, max_tokens=max_tokens)
        except Exception as exc:  # network / provider / parse exception
            last_error = f"LLM call failed: {exc}"
        else:
            if payload is None:
                last_error = "the response contained no parseable JSON object"
            else:
                try:
                    return model_cls.model_validate(payload)
                except ValidationError as exc:
                    last_error = exc

        logger.warning(
            "structured output attempt %s/%s failed for %s: %s",
            attempt, attempts, model_cls.__name__, last_error,
        )
        if attempt >= attempts:
            break

        if payload is not None:
            messages.append({
                "role": "assistant",
                "content": json.dumps(payload, ensure_ascii=False),
            })
        messages.append({
            "role": "user",
            "content": (
                "The previous output was invalid:\n"
                f"{last_error}\n\n"
                "Return a corrected JSON object only. Fix every field named in "
                "the error; do not add extra fields."
            ),
        })

    raise StructuredOutputError(
        f"{model_cls.__name__} failed after {attempts} attempt(s)",
        attempts=attempts,
        last_error=last_error,
    )
