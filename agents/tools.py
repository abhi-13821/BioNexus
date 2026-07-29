"""
agents/tools.py

Shared utility functions for the BioNexus Multi-Agent AI System.

This module provides reusable helper functions that all agents can share.
It includes text processing, response formatting, validation, retry logic,
timing decorators, and other common utilities.

Design decisions:
    - All functions are pure and stateless unless explicitly documented
    - No biomedical business logic is implemented here
    - Functions are reusable across all agents
    - Complete type hints for IDE support
    - Production-quality error handling

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# ----------------------------------------------------------------------
# Text processing utilities
# ----------------------------------------------------------------------


def clean_text(text: Any) -> str:
    """
    Clean and normalize text.

    Args:
        text: The text to clean.

    Returns:
        Cleaned text string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: The text to truncate.
        max_length: Maximum length.
        suffix: Suffix to append if truncated.

    Returns:
        Truncated text.
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> list[str]:
    """
    Extract keywords from text.

    Args:
        text: The text to extract keywords from.
        min_length: Minimum keyword length.
        max_keywords: Maximum number of keywords to return.

    Returns:
        A list of keywords.
    """
    if not text:
        return []

    # Clean and lowercase
    text = clean_text(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Split into words
    words = text.split()

    # Filter by length and common stopwords
    stopwords = {
        "a", "an", "the", "and", "or", "but", "if", "then", "so", "of",
        "in", "on", "at", "to", "for", "with", "by", "from", "as", "is",
        "are", "was", "were", "be", "been", "being", "this", "that",
        "these", "those", "we", "you", "they", "he", "she", "it", "i",
    }

    keywords = []
    for word in words:
        if len(word) >= min_length and word not in stopwords:
            keywords.append(word)

    # Count frequencies
    freq: dict[str, int] = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency
    sorted_keywords = sorted(freq.items(), key=lambda x: -x[1])

    return [word for word, _ in sorted_keywords[:max_keywords]]


def extract_entities_from_text(text: str) -> dict[str, list[str]]:
    """
    Extract potential biomedical entities from text.

    This is a lightweight, regex-based entity extractor. It is not
    a substitute for a proper NER model but provides a quick way to
    surface potential entities.

    Args:
        text: The text to extract entities from.

    Returns:
        A dictionary of entity types to lists of entities.
    """
    if not text:
        return {}

    entities: dict[str, list[str]] = {
        "genes": [],
        "proteins": [],
        "drugs": [],
        "diseases": [],
    }

    # Gene/protein patterns (e.g., BRCA1, EGFR, TP53)
    gene_pattern = r"\b[A-Z][A-Z0-9]{1,7}\d?\b"
    gene_matches = re.findall(gene_pattern, text)
    # Filter out common false positives
    gene_matches = [g for g in gene_matches if len(g) >= 3 and not g.isdigit()]
    entities["genes"] = list(set(gene_matches))[:10]

    # Drug patterns (common suffixes)
    drug_suffixes = r"(mab|nib|prazole|statin|cillin|azole|olol|pril|sartan|vir|umab|ximab)"
    drug_pattern = rf"\b[A-Za-z]{{3,}}{drug_suffixes}\b"
    drug_matches = re.findall(drug_pattern, text, re.IGNORECASE)
    entities["drugs"] = list(set(drug_matches))[:10]

    # Disease patterns (common disease indicators)
    disease_indicators = [
        "cancer", "carcinoma", "diabetes", "alzheimer", "parkinson",
        "syndrome", "disease", "disorder", "infection", "arthritis",
        "asthma", "stroke", "obesity", "hypertension", "depression",
        "sclerosis", "melanoma", "leukemia", "lymphoma", "tumor",
    ]
    text_lower = text.lower()
    diseases = []
    for indicator in disease_indicators:
        if indicator in text_lower:
            # Find the actual term with surrounding context
            pattern = rf"\b[\w\s]{{0,20}}{indicator}[\w\s]{{0,20}}\b"
            matches = re.findall(pattern, text, re.IGNORECASE)
            diseases.extend([m.strip() for m in matches[:5]])
    entities["diseases"] = list(set(diseases))[:10]

    # Remove empty lists
    return {k: v for k, v in entities.items() if v}


# ----------------------------------------------------------------------
# Response formatting utilities
# ----------------------------------------------------------------------


def format_response(
    content: str,
    metadata: Optional[dict[str, Any]] = None,
    confidence: Optional[float] = None,
) -> dict[str, Any]:
    """
    Format a response with metadata.

    Args:
        content: The response content.
        metadata: Additional metadata.
        confidence: Confidence score.

    Returns:
        A formatted response dictionary.
    """
    response: dict[str, Any] = {
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if metadata:
        response["metadata"] = metadata

    if confidence is not None:
        response["confidence"] = min(1.0, max(0.0, confidence))

    return response


def format_error_response(
    error: Union[str, Exception],
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Format an error response.

    Args:
        error: The error message or exception.
        context: Additional context.

    Returns:
        A formatted error response dictionary.
    """
    if isinstance(error, Exception):
        error_msg = str(error)
        error_type = type(error).__name__
    else:
        error_msg = str(error)
        error_type = "Error"

    response: dict[str, Any] = {
        "success": False,
        "error": error_msg,
        "error_type": error_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if context:
        response["context"] = context

    return response


def format_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Format a success response.

    Args:
        data: The response data.
        message: Optional success message.
        metadata: Additional metadata.

    Returns:
        A formatted success response dictionary.
    """
    response: dict[str, Any] = {
        "success": True,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if message:
        response["message"] = message

    if metadata:
        response["metadata"] = metadata

    return response


def to_json_safe(obj: Any) -> Any:
    """
    Convert an object to a JSON-safe representation.

    Args:
        obj: The object to convert.

    Returns:
        A JSON-safe representation.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return to_json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return to_json_safe(obj.__dict__)
    return str(obj)


def pretty_json(obj: Any, indent: int = 2) -> str:
    """
    Convert an object to pretty-printed JSON.

    Args:
        obj: The object to convert.
        indent: Indentation level.

    Returns:
        A pretty-printed JSON string.
    """
    return json.dumps(to_json_safe(obj), indent=indent, default=str)


# ----------------------------------------------------------------------
# Confidence score utilities
# ----------------------------------------------------------------------


def normalize_confidence(score: float) -> float:
    """
    Normalize a confidence score to [0.0, 1.0].

    Args:
        score: The score to normalize.

    Returns:
        A normalized score.
    """
    return min(1.0, max(0.0, score))


def aggregate_confidence(scores: list[float], method: str = "mean") -> float:
    """
    Aggregate multiple confidence scores.

    Args:
        scores: List of confidence scores.
        method: Aggregation method ('mean', 'max', 'min', 'product').

    Returns:
        Aggregated confidence score.

    Raises:
        ValueError: If the method is unknown.
    """
    if not scores:
        return 0.0

    normalized = [normalize_confidence(s) for s in scores]

    if method == "mean":
        return sum(normalized) / len(normalized)
    elif method == "max":
        return max(normalized)
    elif method == "min":
        return min(normalized)
    elif method == "product":
        result = 1.0
        for s in normalized:
            result *= s
        return result
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def confidence_to_label(score: float) -> str:
    """
    Convert a confidence score to a qualitative label.

    Args:
        score: The confidence score.

    Returns:
        A qualitative label: 'low', 'medium', 'high', 'very_high'.
    """
    s = normalize_confidence(score)

    if s >= 0.9:
        return "very_high"
    elif s >= 0.7:
        return "high"
    elif s >= 0.4:
        return "medium"
    else:
        return "low"


# ----------------------------------------------------------------------
# Validation utilities
# ----------------------------------------------------------------------


def validate_non_empty(value: Any, field_name: str) -> bool:
    """
    Validate that a value is non-empty.

    Args:
        value: The value to validate.
        field_name: The field name for error messages.

    Returns:
        True if valid.

    Raises:
        ValueError: If the value is empty.
    """
    if value is None:
        raise ValueError(f"{field_name} must not be None")

    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} must not be empty")

    if isinstance(value, (list, tuple, set, dict)) and not value:
        raise ValueError(f"{field_name} must not be empty")

    return True


def validate_type(value: Any, expected_type: type, field_name: str) -> bool:
    """
    Validate that a value is of the expected type.

    Args:
        value: The value to validate.
        expected_type: The expected type.
        field_name: The field name for error messages.

    Returns:
        True if valid.

    Raises:
        TypeError: If the value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be of type {expected_type.__name__}")
    return True


def validate_uuid(value: str, field_name: str) -> bool:
    """
    Validate that a string is a valid UUID.

    Args:
        value: The string to validate.
        field_name: The field name for error messages.

    Returns:
        True if valid.

    Raises:
        ValueError: If the string is not a valid UUID.
    """
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        raise ValueError(f"{field_name} must be a valid UUID")


def validate_positive(value: Union[int, float], field_name: str) -> bool:
    """
    Validate that a value is positive.

    Args:
        value: The value to validate.
        field_name: The field name for error messages.

    Returns:
        True if valid.

    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return True


def validate_in_range(value: float, min_val: float, max_val: float, field_name: str) -> bool:
    """
    Validate that a value is within a range.

    Args:
        value: The value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        field_name: The field name for error messages.

    Returns:
        True if valid.

    Raises:
        ValueError: If the value is outside the range.
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
    return True


# ----------------------------------------------------------------------
# Retry utilities
# ----------------------------------------------------------------------


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retries.
        delay: Initial delay in seconds.
        backoff: Backoff multiplier.
        exceptions: Tuple of exceptions to catch.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function.

    Raises:
        Exception: The last exception raised.
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise

            logger.warning(
                "Retry attempt %d/%d for %s: %s",
                attempt + 1,
                max_retries,
                func.__name__,
                e,
            )
            await asyncio.sleep(current_delay)
            current_delay *= backoff

    # Should never reach here
    raise last_exception  # type: ignore


def retry_sync(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Retry a synchronous function with exponential backoff.

    Args:
        func: The function to retry.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retries.
        delay: Initial delay in seconds.
        backoff: Backoff multiplier.
        exceptions: Tuple of exceptions to catch.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function.

    Raises:
        Exception: The last exception raised.
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise

            logger.warning(
                "Retry attempt %d/%d for %s: %s",
                attempt + 1,
                max_retries,
                func.__name__,
                e,
            )
            time.sleep(current_delay)
            current_delay *= backoff

    # Should never reach here
    raise last_exception  # type: ignore


# ----------------------------------------------------------------------
# Timing decorators
# ----------------------------------------------------------------------


def timed(label: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator that logs the execution time of a function.

    Args:
        label: Optional label for the log message.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                name = label or func.__name__
                logger.debug("%s completed in %.4f seconds", name, elapsed)

        return wrapper  # type: ignore

    return decorator


def async_timed(label: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator that logs the execution time of an async function.

    Args:
        label: Optional label for the log message.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                name = label or func.__name__
                logger.debug("%s completed in %.4f seconds", name, elapsed)

        return wrapper  # type: ignore

    return decorator


# ----------------------------------------------------------------------
# ID generation utilities
# ----------------------------------------------------------------------


def generate_id(prefix: Optional[str] = None) -> str:
    """
    Generate a unique ID.

    Args:
        prefix: Optional prefix for the ID.

    Returns:
        A unique ID string.
    """
    uid = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def generate_hash(*args: Any) -> str:
    """
    Generate a deterministic hash from values.

    Args:
        *args: Values to hash.

    Returns:
        A hexadecimal hash string.
    """
    parts = []
    for value in args:
        if isinstance(value, Enum):
            parts.append(f"{type(value).__name__}:{value.value}")
        elif is_dataclass(value):
            parts.append(to_json_safe(value))
        else:
            parts.append(str(value))

    combined = "||".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# Result merging utilities
# ----------------------------------------------------------------------


def merge_dicts(base: dict[str, Any], override: dict[str, Any], deep: bool = True) -> dict[str, Any]:
    """
    Merge two dictionaries.

    Args:
        base: The base dictionary.
        override: The override dictionary.
        deep: Whether to perform deep merging.

    Returns:
        A merged dictionary.
    """
    result = dict(base)

    for key, value in override.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge multiple result dictionaries.

    Args:
        results: List of result dictionaries.

    Returns:
        A merged dictionary.
    """
    merged: dict[str, Any] = {
        "total": len(results),
        "successful": 0,
        "failed": 0,
        "results": [],
        "errors": [],
    }

    for result in results:
        if result.get("success", False):
            merged["successful"] += 1
            merged["results"].append(result.get("data"))
        else:
            merged["failed"] += 1
            merged["errors"].append(result.get("error", "Unknown error"))

    return merged


# ----------------------------------------------------------------------
# Logging utilities
# ----------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the BioNexus naming convention.

    Args:
        name: The logger name.

    Returns:
        A configured logger.
    """
    logger_name = f"bionexus.agents.{name}"
    return logging.getLogger(logger_name)


def log_execution(
    logger: logging.Logger,
    func_name: str,
    args: Optional[dict[str, Any]] = None,
    result: Optional[Any] = None,
    error: Optional[Exception] = None,
    level: int = logging.DEBUG,
) -> None:
    """
    Log function execution details.

    Args:
        logger: The logger to use.
        func_name: The function name.
        args: Function arguments.
        result: The function result.
        error: The exception if any.
        level: The log level.
    """
    if error:
        logger.log(
            level,
            "%s failed with error: %s",
            func_name,
            error,
        )
    else:
        logger.log(
            level,
            "%s completed successfully",
            func_name,
        )


def log_step(step_name: str) -> Callable[[F], F]:
    """
    Decorator that logs function entry and exit.

    Args:
        step_name: The name of the step.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug("Step %s starting...", step_name)
            try:
                result = func(*args, **kwargs)
                logger.debug("Step %s completed successfully", step_name)
                return result
            except Exception as e:
                logger.error("Step %s failed: %s", step_name, e)
                raise

        return wrapper  # type: ignore

    return decorator


# ----------------------------------------------------------------------
# Conversation utilities
# ----------------------------------------------------------------------


def get_conversation_id(user_id: Optional[str], session_id: Optional[str] = None) -> str:
    """
    Get or generate a conversation ID.

    Args:
        user_id: The user ID.
        session_id: The session ID.

    Returns:
        A conversation ID string.
    """
    if session_id:
        return session_id
    if user_id:
        return f"conv_{user_id}_{generate_id()}"
    return f"conv_{generate_id()}"


def summarize_conversation(messages: list[dict[str, Any]], max_turns: int = 10) -> str:
    """
    Summarize a conversation history.

    Args:
        messages: List of conversation messages.
        max_turns: Maximum turns to include.

    Returns:
        A summary string.
    """
    if not messages:
        return "No conversation history."

    # Take the last N turns
    turns = messages[-max_turns * 2:]

    summary = []
    for msg in turns:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate content
        content = truncate_text(content, 100)
        summary.append(f"{role}: {content}")

    return "\n".join(summary)


# ----------------------------------------------------------------------
# Platform detection
# ----------------------------------------------------------------------


def is_streamlit_running() -> bool:
    """
    Check if running in a Streamlit environment.

    Returns:
        True if running in Streamlit.
    """
    try:
        import streamlit as st

        # Check if streamlit runtime is available
        return hasattr(st, "runtime")
    except ImportError:
        return False


def get_environment() -> str:
    """
    Get the current environment.

    Returns:
        'streamlit', 'jupyter', 'cli', or 'unknown'.
    """
    if is_streamlit_running():
        return "streamlit"

    try:
        # Check for Jupyter
        from IPython import get_ipython

        if get_ipython() is not None:
            return "jupyter"
    except ImportError:
        pass

    return "cli"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__: list[str] = [
    # Text processing
    "clean_text",
    "truncate_text",
    "extract_keywords",
    "extract_entities_from_text",
    # Response formatting
    "format_response",
    "format_error_response",
    "format_success_response",
    "to_json_safe",
    "pretty_json",
    # Confidence utilities
    "normalize_confidence",
    "aggregate_confidence",
    "confidence_to_label",
    # Validation
    "validate_non_empty",
    "validate_type",
    "validate_uuid",
    "validate_positive",
    "validate_in_range",
    # Retry
    "retry_async",
    "retry_sync",
    # Timing
    "timed",
    "async_timed",
    # ID generation
    "generate_id",
    "generate_hash",
    # Result merging
    "merge_dicts",
    "merge_results",
    # Logging
    "get_logger",
    "log_execution",
    "log_step",
    # Conversation
    "get_conversation_id",
    "summarize_conversation",
    # Platform
    "is_streamlit_running",
    "get_environment",
]