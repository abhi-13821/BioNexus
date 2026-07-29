"""
drug_discovery/utils.py

Shared utility helpers for the BioNexus Drug Discovery module.

This module centralizes small, dependency-free helper functions that are
reused across the other Drug Discovery modules -- ``molecular_properties``,
``compound_similarity``, ``target_prediction``, ``toxicity_prediction``,
``drug_repurposing``, ``candidate_ranking``, and ``cache``. Consolidating
these helpers here keeps the rest of the module focused on its domain
logic and eliminates duplicate implementations of common concerns such as
input validation, score normalization, safe arithmetic, hashing, timing,
and retries.

Responsibilities:
    - Lightweight SMILES sanity checking (a cheap pre-filter; authoritative
      RDKit-backed validation still lives in ``molecular_properties.py``).
    - General-purpose input validation (non-empty strings, numeric ranges,
      probabilities, positive integers, membership checks).
    - Human-readable formatting of molecular weights, similarity scores,
      and percentages.
    - Score clamping/normalization and confidence-bucket conversion,
      compatible with ``drug_discovery.models`` enums.
    - Safe arithmetic helpers that avoid raising on degenerate inputs
      (division by zero, log/sqrt of non-positive numbers).
    - Basic descriptive statistics helpers.
    - Deterministic content hashing and unique identifier generation.
    - UTC date/time helpers for consistent timestamping.
    - Configuration helpers (deep dict merging, boolean coercion).
    - Serialization helpers that safely handle dataclasses and enums.
    - A performance-timing decorator and a configurable retry decorator.
    - Consistent error-message formatting.
    - Small collection helpers (chunking, de-duplication, truncation) that
      show up repeatedly across the other modules.

Design notes
------------
- This module has no dependency on RDKit or any other Drug Discovery
  module other than ``drug_discovery.models`` (for enum/type
  interoperability), so it can be imported freely without introducing
  circular imports.
- All functions are pure and stateless unless explicitly documented
  otherwise (e.g., the timing/retry decorators wrap arbitrary callables
  but do not themselves hold mutable state beyond their closures).

Compatibility
-------------
Targets Python 3.11. Uses only the standard library plus
``drug_discovery.models``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import re
import statistics
import time
import uuid
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from drug_discovery.models import PredictionConfidence

__all__ = [
    "UtilityError",
    "InputValidationError",
    "RetryExhaustedError",
    "is_plausible_smiles",
    "validate_non_empty_string",
    "validate_in_range",
    "validate_probability",
    "validate_positive_integer",
    "validate_positive_number",
    "validate_one_of",
    "format_molecular_weight",
    "format_similarity_score",
    "format_percentage",
    "clamp",
    "normalize_min_max",
    "score_to_confidence",
    "confidence_to_midpoint_score",
    "safe_divide",
    "safe_log",
    "safe_sqrt",
    "safe_mean",
    "BasicStatistics",
    "compute_basic_statistics",
    "compute_percentile",
    "generate_content_hash",
    "generate_unique_id",
    "utc_now",
    "format_iso_timestamp",
    "parse_iso_timestamp",
    "seconds_since",
    "deep_merge_dicts",
    "coerce_bool",
    "safe_json_dumps",
    "safe_json_loads",
    "timed",
    "retry",
    "format_exception_message",
    "summarize_exception_chain",
    "chunked",
    "deduplicate_preserving_order",
    "truncate_text",
]

logger = logging.getLogger(__name__)

_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class UtilityError(RuntimeError):
    """Base exception for unexpected failures within utility helpers."""


class InputValidationError(ValueError):
    """
    Raised when a general-purpose input validation helper rejects a
    value.

    Attributes:
        field_name: The name of the field or parameter that failed
            validation, used to produce a clear error message.
    """

    def __init__(self, field_name: str, message: str) -> None:
        self.field_name = field_name
        super().__init__(message)


class RetryExhaustedError(RuntimeError):
    """
    Raised by the ``retry`` decorator when all configured attempts have
    been exhausted without a successful call.

    Attributes:
        attempts: The total number of attempts that were made.
        last_exception: The exception raised by the final failed attempt.
    """

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempt(s); "
            f"last error: {last_exception!r}"
        )


# ---------------------------------------------------------------------------
# Lightweight SMILES sanity checking
# ---------------------------------------------------------------------------

# A permissive character-class check covering standard SMILES syntax
# (atoms, bonds, ring closures, branches, charges, isotopes, stereo
# markers, and organic subset shorthand). This is intentionally NOT a
# substitute for RDKit parsing -- it exists as a cheap, dependency-free
# pre-filter that callers can use to reject obviously malformed input
# before invoking the more expensive RDKit-backed validation in
# ``molecular_properties.validate_smiles``.
_SMILES_ALLOWED_PATTERN = re.compile(
    r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%~]+$"
)


def is_plausible_smiles(smiles: str) -> bool:
    """
    Perform a cheap, dependency-free structural sanity check on a SMILES
    string.

    This checks that the string is non-empty, contains only characters
    that can legally appear in SMILES notation, and has balanced
    parentheses and square brackets. It does NOT verify chemical
    validity (valence, aromaticity, ring closure correctness, etc.) --
    use ``molecular_properties.validate_smiles`` for authoritative
    RDKit-backed validation.

    Args:
        smiles: The candidate SMILES string.

    Returns:
        True if the string passes basic structural sanity checks, False
        otherwise.
    """
    if not isinstance(smiles, str):
        return False

    candidate = smiles.strip()
    if not candidate:
        return False

    if not _SMILES_ALLOWED_PATTERN.match(candidate):
        return False

    paren_balance = 0
    bracket_balance = 0
    for char in candidate:
        if char == "(":
            paren_balance += 1
        elif char == ")":
            paren_balance -= 1
        elif char == "[":
            bracket_balance += 1
        elif char == "]":
            bracket_balance -= 1
        if paren_balance < 0 or bracket_balance < 0:
            return False

    return paren_balance == 0 and bracket_balance == 0


# ---------------------------------------------------------------------------
# General input validation
# ---------------------------------------------------------------------------


def validate_non_empty_string(value: Any, field_name: str) -> str:
    """
    Validate that a value is a non-empty (after stripping) string.

    Args:
        value: The value to validate.
        field_name: The name of the field being validated, used in error
            messages.

    Returns:
        The stripped string value.

    Raises:
        InputValidationError: If ``value`` is not a string, or is empty
            after stripping.
    """
    if not isinstance(value, str):
        raise InputValidationError(
            field_name, f"'{field_name}' must be a string, got {type(value).__name__}."
        )
    stripped = value.strip()
    if not stripped:
        raise InputValidationError(field_name, f"'{field_name}' must be a non-empty string.")
    return stripped


def validate_in_range(
    value: float, minimum: float, maximum: float, field_name: str
) -> float:
    """
    Validate that a numeric value falls within an inclusive range.

    Args:
        value: The value to validate.
        minimum: The inclusive lower bound.
        maximum: The inclusive upper bound.
        field_name: The name of the field being validated, used in error
            messages.

    Returns:
        The validated value, unchanged.

    Raises:
        InputValidationError: If ``value`` is not numeric or falls
            outside ``[minimum, maximum]``.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InputValidationError(
            field_name, f"'{field_name}' must be numeric, got {type(value).__name__}."
        )
    if not minimum <= value <= maximum:
        raise InputValidationError(
            field_name,
            f"'{field_name}' must be within [{minimum}, {maximum}], got {value}.",
        )
    return value


def validate_probability(value: float, field_name: str) -> float:
    """
    Validate that a numeric value is a valid probability/score in
    [0.0, 1.0].

    Args:
        value: The value to validate.
        field_name: The name of the field being validated, used in error
            messages.

    Returns:
        The validated value, unchanged.

    Raises:
        InputValidationError: If ``value`` is not numeric or falls
            outside ``[0.0, 1.0]``.
    """
    return validate_in_range(value, 0.0, 1.0, field_name)


def validate_positive_integer(value: Any, field_name: str) -> int:
    """
    Validate that a value is a strictly positive integer.

    Args:
        value: The value to validate.
        field_name: The name of the field being validated, used in error
            messages.

    Returns:
        The validated integer value.

    Raises:
        InputValidationError: If ``value`` is not an integer or is not
            positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(
            field_name, f"'{field_name}' must be an integer, got {type(value).__name__}."
        )
    if value < 1:
        raise InputValidationError(field_name, f"'{field_name}' must be positive, got {value}.")
    return value


def validate_positive_number(value: Any, field_name: str) -> float:
    """
    Validate that a value is a strictly positive number (int or float).

    Args:
        value: The value to validate.
        field_name: The name of the field being validated, used in error
            messages.

    Returns:
        The validated value as a float.

    Raises:
        InputValidationError: If ``value`` is not numeric or is not
            positive.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InputValidationError(
            field_name, f"'{field_name}' must be numeric, got {type(value).__name__}."
        )
    if value <= 0:
        raise InputValidationError(field_name, f"'{field_name}' must be positive, got {value}.")
    return float(value)


def validate_one_of(value: Any, allowed: Sequence[Any], field_name: str) -> Any:
    """
    Validate that a value is a member of an allowed set of values.

    Args:
        value: The value to validate.
        allowed: The sequence of permitted values.
        field_name: The name of the field being validated, used in error
            messages.

    Returns:
        The validated value, unchanged.

    Raises:
        InputValidationError: If ``value`` is not present in ``allowed``.
    """
    if value not in allowed:
        raise InputValidationError(
            field_name,
            f"'{field_name}' must be one of {list(allowed)}, got {value!r}.",
        )
    return value


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_molecular_weight(
    molecular_weight: float, unit: str = "g/mol", precision: int = 2
) -> str:
    """
    Format a molecular weight value for human-readable display.

    Args:
        molecular_weight: The molecular weight value.
        unit: The unit label to append (default ``"g/mol"``).
        precision: Number of decimal places to display.

    Returns:
        A formatted string, e.g. ``"180.16 g/mol"``.

    Raises:
        InputValidationError: If ``molecular_weight`` is negative.
    """
    validate_in_range(molecular_weight, 0.0, math.inf, "molecular_weight")
    return f"{molecular_weight:.{precision}f} {unit}".strip()


def format_similarity_score(
    score: float, metric: Any = None, precision: int = 3
) -> str:
    """
    Format a similarity score for human-readable display, optionally
    annotated with the metric used to compute it.

    Args:
        score: The similarity score, expected in [0.0, 1.0].
        metric: Optional metric identifier (e.g., a
            ``drug_discovery.models.SimilarityMetric`` member, or any
            object with a meaningful ``str()``/``.value``).
        precision: Number of decimal places to display.

    Returns:
        A formatted string, e.g. ``"0.847 (tanimoto)"`` or ``"0.847"``
        if no metric is supplied.
    """
    validate_probability(score, "score")
    formatted = f"{score:.{precision}f}"
    if metric is None:
        return formatted
    metric_label = metric.value if isinstance(metric, Enum) else str(metric)
    return f"{formatted} ({metric_label})"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a fractional value in [0.0, 1.0] as a percentage string.

    Args:
        value: The fractional value to format.
        precision: Number of decimal places to display.

    Returns:
        A formatted percentage string, e.g. ``"84.7%"``.
    """
    validate_probability(value, "value")
    return f"{value * 100:.{precision}f}%"


# ---------------------------------------------------------------------------
# Score normalization and confidence conversion
# ---------------------------------------------------------------------------


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clamp a numeric value into a closed interval.

    Args:
        value: The value to clamp.
        minimum: The inclusive lower bound.
        maximum: The inclusive upper bound.

    Returns:
        ``value`` restricted to ``[minimum, maximum]``.
    """
    return min(maximum, max(minimum, value))


def normalize_min_max(value: float, minimum: float, maximum: float) -> float:
    """
    Linearly rescale a value from an arbitrary range onto [0.0, 1.0].

    Args:
        value: The raw value to rescale.
        minimum: The lower bound of the source range.
        maximum: The upper bound of the source range.

    Returns:
        The rescaled value in [0.0, 1.0]. If ``minimum == maximum``, the
        midpoint value ``0.5`` is returned, since no differentiating
        range information is available.
    """
    if maximum == minimum:
        return 0.5
    return clamp((value - minimum) / (maximum - minimum))


def score_to_confidence(score: float) -> PredictionConfidence:
    """
    Convert a numeric confidence-like score in [0.0, 1.0] into a
    qualitative ``PredictionConfidence`` bucket.

    Uses the same bucket boundaries applied elsewhere in the Drug
    Discovery module (e.g., ``candidate_ranking``): scores at or above
    0.85 are ``VERY_HIGH``, at or above 0.65 are ``HIGH``, at or above
    0.4 are ``MEDIUM``, and everything else is ``LOW``.

    Args:
        score: A numeric score in [0.0, 1.0].

    Returns:
        The corresponding ``PredictionConfidence`` bucket.

    Raises:
        InputValidationError: If ``score`` is outside [0.0, 1.0].
    """
    validate_probability(score, "score")
    if score >= 0.85:
        return PredictionConfidence.VERY_HIGH
    if score >= 0.65:
        return PredictionConfidence.HIGH
    if score >= 0.4:
        return PredictionConfidence.MEDIUM
    return PredictionConfidence.LOW


def confidence_to_midpoint_score(confidence: PredictionConfidence) -> float:
    """
    Convert a qualitative ``PredictionConfidence`` bucket into a
    representative numeric midpoint score.

    Useful when a downstream computation (e.g., aggregate statistics)
    needs a numeric proxy for a qualitative confidence bucket.

    Args:
        confidence: The confidence bucket to convert.

    Returns:
        A representative score in [0.0, 1.0]: 0.15 for LOW, 0.5 for
        MEDIUM, 0.75 for HIGH, and 0.925 for VERY_HIGH.

    Raises:
        InputValidationError: If ``confidence`` is not a recognized
            ``PredictionConfidence`` member.
    """
    midpoints = {
        PredictionConfidence.LOW: 0.15,
        PredictionConfidence.MEDIUM: 0.5,
        PredictionConfidence.HIGH: 0.75,
        PredictionConfidence.VERY_HIGH: 0.925,
    }
    if confidence not in midpoints:
        raise InputValidationError(
            "confidence", f"Unrecognized PredictionConfidence value: {confidence!r}."
        )
    return midpoints[confidence]


# ---------------------------------------------------------------------------
# Safe arithmetic helpers
# ---------------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide two numbers, returning a default value instead of raising on
    division by zero.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: The value to return if ``denominator`` is zero.

    Returns:
        ``numerator / denominator``, or ``default`` if ``denominator``
        is zero.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def safe_log(value: float, base: float = math.e, default: float = 0.0) -> float:
    """
    Compute a logarithm, returning a default value instead of raising for
    non-positive input.

    Args:
        value: The value to take the logarithm of.
        base: The logarithm base (defaults to the natural log base).
        default: The value to return if ``value`` is not strictly
            positive.

    Returns:
        ``log(value, base)``, or ``default`` if ``value <= 0``.
    """
    if value <= 0:
        return default
    return math.log(value, base)


def safe_sqrt(value: float, default: float = 0.0) -> float:
    """
    Compute a square root, returning a default value instead of raising
    for negative input.

    Args:
        value: The value to take the square root of.
        default: The value to return if ``value`` is negative.

    Returns:
        ``sqrt(value)``, or ``default`` if ``value < 0``.
    """
    if value < 0:
        return default
    return math.sqrt(value)


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    """
    Compute the arithmetic mean of a sequence, returning a default value
    instead of raising for an empty sequence.

    Args:
        values: The sequence of numeric values.
        default: The value to return if ``values`` is empty.

    Returns:
        The arithmetic mean of ``values``, or ``default`` if empty.
    """
    if not values:
        return default
    return statistics.mean(values)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


class BasicStatistics:
    """
    Container for a compact set of descriptive statistics computed over
    a sequence of numeric values.

    Attributes:
        count: Number of values summarized.
        mean: Arithmetic mean.
        median: Median value.
        std_dev: Population standard deviation (0.0 if fewer than two
            values).
        minimum: Minimum observed value.
        maximum: Maximum observed value.
    """

    __slots__ = ("count", "mean", "median", "std_dev", "minimum", "maximum")

    def __init__(
        self,
        count: int,
        mean: float,
        median: float,
        std_dev: float,
        minimum: float,
        maximum: float,
    ) -> None:
        self.count = count
        self.mean = mean
        self.median = median
        self.std_dev = std_dev
        self.minimum = minimum
        self.maximum = maximum

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"BasicStatistics(count={self.count}, mean={self.mean:.4f}, "
            f"median={self.median:.4f}, std_dev={self.std_dev:.4f}, "
            f"minimum={self.minimum:.4f}, maximum={self.maximum:.4f})"
        )


def compute_basic_statistics(values: Sequence[float]) -> BasicStatistics:
    """
    Compute a compact set of descriptive statistics for a sequence of
    numeric values.

    Args:
        values: The sequence of numeric values to summarize.

    Returns:
        A populated ``BasicStatistics`` instance. All fields are zero if
        ``values`` is empty.
    """
    if not values:
        return BasicStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return BasicStatistics(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        std_dev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    """
    Compute the value at a given percentile within a sequence, using
    linear interpolation between closest ranks.

    Args:
        values: The sequence of numeric values.
        percentile: The desired percentile, in [0.0, 100.0].

    Returns:
        The interpolated value at the requested percentile. Returns 0.0
        for an empty sequence.

    Raises:
        InputValidationError: If ``percentile`` is outside [0.0, 100.0].
    """
    validate_in_range(percentile, 0.0, 100.0, "percentile")
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[int(rank)]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = rank - lower_index
    return lower_value + (upper_value - lower_value) * fraction


# ---------------------------------------------------------------------------
# Hashing and identifier generation
# ---------------------------------------------------------------------------


def generate_content_hash(*values: Any) -> str:
    """
    Generate a deterministic SHA-256 hex digest for an arbitrary set of
    values.

    Values are converted to a stable string representation before
    hashing, so the same logical inputs always produce the same hash
    regardless of process. This is intended for cache-adjacent uses such
    as deduplication keys or content fingerprints; for cache key
    generation itself, prefer ``cache.make_cache_key``.

    Args:
        *values: The values to hash. Each is converted via ``repr()``
            after JSON-normalizing primitives, enums, and dataclasses
            where possible.

    Returns:
        A hexadecimal SHA-256 digest string.
    """
    parts: list[str] = []
    for value in values:
        if isinstance(value, Enum):
            parts.append(f"enum:{type(value).__name__}={value.value!r}")
        elif is_dataclass(value) and not isinstance(value, type):
            parts.append(f"dataclass:{type(value).__name__}={asdict(value)!r}")
        else:
            parts.append(repr(value))
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_unique_id(prefix: str | None = None) -> str:
    """
    Generate a unique identifier suitable for run IDs, candidate IDs, or
    other record identifiers.

    Args:
        prefix: Optional prefix to prepend, separated by an underscore
            (e.g., ``"RUN"`` yields ``"RUN_3fa85f64...")``.

    Returns:
        A unique identifier string based on a UUID4.
    """
    identifier = uuid.uuid4().hex
    return f"{prefix}_{identifier}" if prefix else identifier


# ---------------------------------------------------------------------------
# Date and time helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """
    Return the current UTC timestamp.

    Returns:
        A timezone-aware ``datetime`` in UTC.
    """
    return datetime.now(timezone.utc)


def format_iso_timestamp(moment: datetime) -> str:
    """
    Format a ``datetime`` as an ISO-8601 string.

    Args:
        moment: The ``datetime`` to format. Naive datetimes are assumed
            to already be in UTC.

    Returns:
        An ISO-8601 formatted string.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat()


def parse_iso_timestamp(value: str) -> datetime:
    """
    Parse an ISO-8601 timestamp string into a timezone-aware ``datetime``.

    Args:
        value: The ISO-8601 formatted timestamp string.

    Returns:
        A timezone-aware ``datetime`` instance (assumed UTC if the input
        string carries no offset).

    Raises:
        InputValidationError: If ``value`` cannot be parsed as an
            ISO-8601 timestamp.
    """
    validate_non_empty_string(value, "value")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputValidationError(
            "value", f"'{value}' is not a valid ISO-8601 timestamp: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def seconds_since(moment: datetime) -> float:
    """
    Compute the number of seconds elapsed since a given timestamp.

    Args:
        moment: The reference ``datetime``. Naive datetimes are assumed
            to already be in UTC.

    Returns:
        The elapsed time in seconds as a float. Never negative; clamped
        to 0.0 if ``moment`` is in the future.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = (utc_now() - moment).total_seconds()
    return max(0.0, delta)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, with values in ``override``
    taking precedence over ``base``.

    Nested dictionaries are merged recursively; all other value types
    (including lists) are replaced wholesale by the ``override`` value.
    Neither input dictionary is mutated.

    Args:
        base: The base dictionary.
        override: The dictionary whose values take precedence.

    Returns:
        A new, merged dictionary.
    """
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge_dicts(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def coerce_bool(value: Any, default: bool = False) -> bool:
    """
    Coerce a loosely-typed value (string, number, or bool) into a
    boolean.

    Recognizes the case-insensitive strings ``"true"``, ``"1"``,
    ``"yes"``, and ``"on"`` as True, and ``"false"``, ``"0"``, ``"no"``,
    and ``"off"`` as False. Falls back to Python truthiness for other
    types, and to ``default`` for ``None``.

    Args:
        value: The value to coerce.
        default: The value to return if ``value`` is ``None``.

    Returns:
        The coerced boolean value.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return bool(normalized)
    return bool(value)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """
    Fallback serializer used by ``safe_json_dumps`` for objects that are
    not natively JSON-serializable.

    Handles ``Enum`` members, dataclass instances, and ``datetime``
    objects; falls back to ``str()`` for anything else.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable representation of ``value``.
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, datetime):
        return format_iso_timestamp(value)
    return str(value)


def safe_json_dumps(obj: Any, indent: int | None = None) -> str:
    """
    Serialize an object to a JSON string, gracefully handling enums,
    dataclasses, and datetimes.

    Args:
        obj: The object to serialize.
        indent: Optional indentation level, forwarded to ``json.dumps``.

    Returns:
        A JSON-formatted string.

    Raises:
        UtilityError: If ``obj`` cannot be serialized even with the
            fallback serializer.
    """
    try:
        return json.dumps(obj, default=_json_default, indent=indent, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise UtilityError(f"Failed to serialize object to JSON: {exc}") from exc


def safe_json_loads(payload: str) -> Any:
    """
    Deserialize a JSON string, raising a domain-specific error on
    failure.

    Args:
        payload: The JSON-formatted string to parse.

    Returns:
        The deserialized Python object.

    Raises:
        UtilityError: If ``payload`` is not valid JSON.
    """
    try:
        return json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise UtilityError(f"Failed to deserialize JSON payload: {exc}") from exc


# ---------------------------------------------------------------------------
# Performance timing decorator
# ---------------------------------------------------------------------------


def timed(
    label: str | None = None,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Decorator that logs (at DEBUG level) the wall-clock execution time of
    the wrapped function.

    Intended for instrumenting the expensive, deterministic computations
    scattered across the Drug Discovery module (descriptor calculation,
    similarity search, target/toxicity prediction, ranking) without
    cluttering their implementations with manual timing code.

    Args:
        label: Optional human-readable label used in the log message. If
            omitted, the wrapped function's qualified name is used.

    Returns:
        A decorator that wraps a function with timing instrumentation.
        The wrapped function gains a ``last_duration_seconds`` attribute
        holding the duration of its most recent call.
    """

    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        effective_label = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                wrapper.last_duration_seconds = elapsed  # type: ignore[attr-defined]
                logger.debug(
                    "'%s' completed in %.4f seconds.", effective_label, elapsed
                )

        wrapper.last_duration_seconds = 0.0  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Decorator that retries a function call on failure, using exponential
    backoff between attempts.

    Intended for wrapping calls to external or otherwise flaky
    dependencies (e.g., network-backed target/repurposing data sources)
    within the Drug Discovery module. Purely local, deterministic
    computations generally should not be wrapped with this decorator.

    Args:
        max_attempts: Maximum number of attempts (including the first),
            must be a positive integer.
        delay_seconds: Initial delay, in seconds, before the first retry.
        backoff_factor: Multiplier applied to the delay after each
            failed attempt.
        exceptions: Tuple of exception types that should trigger a
            retry. Exceptions not in this tuple propagate immediately.

    Returns:
        A decorator that wraps a function with retry behavior.

    Raises:
        InputValidationError: If ``max_attempts`` is not a positive
            integer.
    """
    validate_positive_integer(max_attempts, "max_attempts")

    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _R:
            current_delay = delay_seconds
            last_exception: BaseException | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exception = exc
                    if attempt >= max_attempts:
                        break
                    logger.warning(
                        "Attempt %d/%d for '%s' failed with %s; retrying in "
                        "%.2fs.",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            assert last_exception is not None  # for type-checkers
            raise RetryExhaustedError(max_attempts, last_exception) from last_exception

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Error formatting helpers
# ---------------------------------------------------------------------------


def format_exception_message(exc: BaseException) -> str:
    """
    Format an exception into a concise, single-line human-readable
    message.

    Args:
        exc: The exception to format.

    Returns:
        A string of the form ``"ExceptionType: message"``.
    """
    return f"{type(exc).__name__}: {exc}"


def summarize_exception_chain(exc: BaseException) -> list[str]:
    """
    Summarize an exception and its full ``__cause__`` chain into a list
    of human-readable messages, outermost first.

    Useful for logging or surfacing detailed error context (e.g., a
    ``CandidateRankingError`` raised from an underlying
    ``DescriptorCalculationError``) without printing a full traceback.

    Args:
        exc: The exception at the head of the chain.

    Returns:
        A list of formatted exception messages, ordered from the given
        exception to its root cause.
    """
    summary: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        summary.append(format_exception_message(current))
        current = current.__cause__
    return summary


# ---------------------------------------------------------------------------
# Common collection helpers
# ---------------------------------------------------------------------------


def chunked(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """
    Split a sequence into consecutive chunks of a given maximum size.

    Useful for batching compound lists before submitting them to
    expensive per-batch operations (e.g., similarity search or target
    prediction over large compound libraries).

    Args:
        items: The sequence to split into chunks.
        size: The maximum size of each chunk, must be a positive
            integer.

    Yields:
        Successive lists of up to ``size`` items each.

    Raises:
        InputValidationError: If ``size`` is not a positive integer.
    """
    validate_positive_integer(size, "size")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def deduplicate_preserving_order(items: Iterable[Any]) -> list[Any]:
    """
    Remove duplicate items from an iterable while preserving the order
    of first occurrence.

    Items must be hashable.

    Args:
        items: The iterable of (hashable) items to deduplicate.

    Returns:
        A list containing only the first occurrence of each distinct
        item, in original order.
    """
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, appending a suffix if
    truncation occurred.

    Useful for keeping generated explanations, notes, or report
    fragments within a bounded display width.

    Args:
        text: The text to truncate.
        max_length: The maximum total length of the returned string
            (including the suffix), must be a positive integer.
        suffix: The suffix to append when truncation occurs.

    Returns:
        The original string if it already fits within ``max_length``,
        otherwise a truncated string ending in ``suffix``.

    Raises:
        InputValidationError: If ``max_length`` is not a positive
            integer.
    """
    validate_positive_integer(max_length, "max_length")
    if len(text) <= max_length:
        return text
    if len(suffix) >= max_length:
        return suffix[:max_length]
    return text[: max_length - len(suffix)] + suffix