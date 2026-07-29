"""
relation_extractor.py
======================

Subject-Verb-Object (SVO) relation extraction layer for the BioNexus
knowledge graph pipeline.

This module wraps a spaCy dependency-parsing pipeline behind a single
stable interface (`RelationExtractor`) that:

    * Never raises an uncaught / built-in exception to the caller.
    * Never crashes on missing models, ``None`` input, or empty strings.
    * Returns structured dictionaries (never raw tuples) so downstream
      modules (graph_builder, query_engine) have a stable contract.
    * Assigns a heuristic confidence score to every extracted relation,
      based on the strength of the grammatical signal.

This file has **no dependency on any other module in this package** and
can be imported, instantiated, and tested completely on its own.

Example
-------
>>> from relation_extractor import RelationExtractor
>>> extractor = RelationExtractor(model_name="en_core_web_sm")
>>> relations = extractor.extract_relations(
...     "Aspirin inhibits COX-1 and reduces inflammation."
... )
>>> for rel in relations:
...     print(rel["subject"], rel["verb"], rel["object"], rel["confidence"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level logger (see entity_extractor.py for rationale -- this is a
# standard logging pattern, not shared mutable state).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions, scoped to this module only. Intentionally not shared
# with entity_extractor.py so this file remains independently importable
# and testable without any package-level coupling.
# ---------------------------------------------------------------------------
class ModelLoadError(Exception):
    """Raised when a spaCy language model cannot be loaded for parsing."""


class RelationExtractionError(Exception):
    """
    Raised when relation extraction fails for a reason unrelated to model
    loading (e.g. an unexpected internal spaCy failure). Malformed input
    itself does NOT raise this -- it is handled gracefully instead.
    """


# Dependency labels that can mark a grammatical subject.
_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})

# Dependency labels that can mark a direct grammatical object.
_DIRECT_OBJECT_DEPS = frozenset({"dobj", "attr", "oprd", "dative"})

# Dependency labels that indicate an object reached through a preposition.
_PREP_OBJECT_DEPS = frozenset({"pobj"})

# Part-of-speech tags eligible to act as the "verb" of a relation.
_VERB_POS = frozenset({"VERB", "AUX"})


class RelationExtractor:
    """
    High-level, fault-tolerant extractor of Subject-Verb-Object relations.

    The extractor walks each sentence's dependency tree looking for verbs
    that have both an identifiable subject and object, and emits one
    structured relation record per match, with a heuristic confidence
    score reflecting how grammatically direct the relation is.

    Attributes
    ----------
    model_name : str
        Name of the spaCy model requested at construction time.
    nlp : Optional[Any]
        The loaded spaCy ``Language`` pipeline object, or ``None`` if
        loading failed and ``strict`` was set to ``False``.
    strict : bool
        If True, a failed model load raises ``ModelLoadError`` immediately.
        If False, the extractor stays alive in a degraded state where
        ``extract_relations`` will return an empty list and log an error.
    """

    def __init__(
        self,
        model_name: str = "en_core_sci_sm",
        strict: bool = True,
    ) -> None:
        """
        Initialize the extractor and attempt to load the requested model.

        Parameters
        ----------
        model_name : str, optional
            The spaCy model to load (default: "en_core_web_sm"). Any
            model with a dependency parser component is suitable; a full
            NER component is not required for this module.
        strict : bool, optional
            If True (default), failure to load the model raises
            ``ModelLoadError``. If False, the extractor is created in a
            degraded state instead of raising.

        Raises
        ------
        ModelLoadError
            If ``strict`` is True and the model cannot be loaded, or if
            the loaded pipeline has no dependency parser component.
        """
        self.model_name: str = model_name if isinstance(model_name, str) else ""
        self.strict: bool = bool(strict)
        self.nlp: Optional[Any] = None
        self._model_loaded: bool = False

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """
        Attempt to import spaCy, load ``self.model_name``, and verify it
        includes a dependency parser component.

        Raises
        ------
        ModelLoadError
            Only when ``self.strict`` is True.
        """
        if not self.model_name:
            message = (
                "No valid model_name was provided to RelationExtractor "
                "(received an empty string or non-string value)."
            )
            logger.error(message)
            if self.strict:
                raise ModelLoadError(message)
            return

        try:
            import spacy  # Imported lazily so this module can be used even
            # when spaCy is not installed (e.g. for error-handling tests).
        except ImportError as exc:
            message = (
                "spaCy is not installed in this environment. Install it "
                "with:  pip install spacy"
            )
            logger.error("%s (original error: %s)", message, exc)
            if self.strict:
                raise ModelLoadError(message) from exc
            return

        try:
            self.nlp = spacy.load(self.model_name)
        except OSError as exc:
            message = (
                f"Model '{self.model_name}' could not be loaded. It is "
                f"likely not installed. Try:  python -m spacy download "
                f"{self.model_name}"
            )
            logger.error("%s (original error: %s)", message, exc)
            if self.strict:
                raise ModelLoadError(message) from exc
            self.nlp = None
            return
        except Exception as exc:  # noqa: BLE001 - final safety net
            message = f"Unexpected error while loading model '{self.model_name}'."
            logger.error("%s (original error: %s)", message, exc)
            if self.strict:
                raise ModelLoadError(message) from exc
            self.nlp = None
            return

        if "parser" not in self.nlp.pipe_names:
            message = (
                f"Model '{self.model_name}' loaded successfully but has no "
                "dependency parser component, which RelationExtractor "
                "requires. Choose a model that includes 'parser' in its "
                "pipeline."
            )
            logger.error(message)
            if self.strict:
                raise ModelLoadError(message)
            self.nlp = None
            return

        self._model_loaded = True
        logger.info("Successfully loaded model '%s' with parser.", self.model_name)

    def is_model_loaded(self) -> bool:
        """
        Report whether a usable dependency-parsing pipeline is loaded.

        Returns
        -------
        bool
            True if a spaCy pipeline with a parser was successfully
            loaded, False otherwise.
        """
        return self._model_loaded and self.nlp is not None

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_relations(self, text: Optional[str]) -> List[Dict[str, Any]]:
        """
        Extract Subject-Verb-Object relations from a piece of text.

        This method never raises on bad input. ``None``, non-string
        values, empty strings, and whitespace-only strings all result in
        an empty list plus a logged warning.

        Parameters
        ----------
        text : Optional[str]
            The input text to analyze. May contain multiple sentences.

        Returns
        -------
        List[Dict[str, Any]]
            A list of structured relation dictionaries, each shaped as::

                {
                    "subject": str,
                    "verb": str,
                    "object": str,
                    "confidence": float,   # in range [0.0, 1.0]
                    "sentence": str,       # the full source sentence
                    "metadata": {
                        "model": str,
                        "passive_voice": bool,
                        "negated": bool,
                    },
                }

            Returns ``[]`` (never ``None``) if extraction could not be
            performed for any reason.

        Raises
        ------
        RelationExtractionError
            Only for genuine internal failures during processing of
            otherwise valid, non-empty text. Bad input never triggers
            this.
        """
        cleaned_text = self._validate_and_clean_text(text)
        if cleaned_text is None:
            return []

        if not self.is_model_loaded():
            logger.error(
                "extract_relations called but no model is loaded "
                "(model_name='%s'). Returning empty result.",
                self.model_name,
            )
            return []

        try:
            doc = self.nlp(cleaned_text)
        except Exception as exc:  # noqa: BLE001
            raise RelationExtractionError(
                f"spaCy pipeline failed while processing text: {exc}"
            ) from exc

        relations: List[Dict[str, Any]] = []
        try:
            sentences = list(doc.sents)
        except Exception as exc:  # noqa: BLE001
            raise RelationExtractionError(
                f"Failed to segment text into sentences: {exc}"
            ) from exc

        for sent in sentences:
            try:
                relations.extend(self._extract_from_sentence(sent))
            except Exception as exc:  # noqa: BLE001
                # One malformed sentence should never abort the whole batch.
                logger.warning(
                    "Skipping sentence due to extraction error: %s | text=%r",
                    exc,
                    getattr(sent, "text", "<unavailable>"),
                )
                continue

        return relations

    def _extract_from_sentence(self, sent: Any) -> List[Dict[str, Any]]:
        """
        Extract all SVO relations found within a single spaCy ``Span``.

        Parameters
        ----------
        sent : Any
            A spaCy sentence ``Span`` (an element of ``doc.sents``).

        Returns
        -------
        List[Dict[str, Any]]
            Relation records found in this sentence. May be empty.
        """
        records: List[Dict[str, Any]] = []

        for token in sent:
            if token.pos_ not in _VERB_POS:
                continue

            subject_token = self._find_subject(token)
            if subject_token is None:
                continue

            object_token, is_prepositional = self._find_object(token)
            if object_token is None:
                continue

            is_passive = subject_token.dep_ == "nsubjpass"
            is_negated = self._is_negated(token)

            confidence = self._compute_confidence(
                is_prepositional=is_prepositional,
                is_passive=is_passive,
                is_negated=is_negated,
            )

            records.append(
                {
                    "subject": self._span_text(subject_token),
                    "verb": token.lemma_ if token.lemma_ else token.text,
                    "object": self._span_text(object_token),
                    "confidence": confidence,
                    "sentence": sent.text.strip(),
                    "metadata": {
                        "model": self.model_name,
                        "passive_voice": is_passive,
                        "negated": is_negated,
                    },
                }
            )

        return records

    # ------------------------------------------------------------------
    # Dependency-tree helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_subject(verb_token: Any) -> Optional[Any]:
        """
        Locate the grammatical subject of a verb token.

        Parameters
        ----------
        verb_token : Any
            A spaCy ``Token`` with ``pos_`` in {"VERB", "AUX"}.

        Returns
        -------
        Optional[Any]
            The subject ``Token``, or ``None`` if no subject is attached
            directly to this verb.
        """
        for child in verb_token.children:
            if child.dep_ in _SUBJECT_DEPS:
                return child
        return None

    @staticmethod
    def _find_object(verb_token: Any) -> "tuple[Optional[Any], bool]":
        """
        Locate the grammatical object of a verb token.

        Direct objects (``dobj``, ``attr``, etc.) are preferred over
        prepositional objects reached via a ``prep`` child, since direct
        objects represent a stronger, less ambiguous grammatical signal.

        Parameters
        ----------
        verb_token : Any
            A spaCy ``Token`` with ``pos_`` in {"VERB", "AUX"}.

        Returns
        -------
        tuple[Optional[Any], bool]
            A tuple of ``(object_token, is_prepositional)``. The object
            token is ``None`` if nothing suitable was found.
        """
        # Prefer a direct object first.
        for child in verb_token.children:
            if child.dep_ in _DIRECT_OBJECT_DEPS:
                return child, False

        # Fall back to walking through a preposition, e.g. "acts on X".
        for child in verb_token.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ in _PREP_OBJECT_DEPS:
                        return grandchild, True

        return None, False

    @staticmethod
    def _is_negated(verb_token: Any) -> bool:
        """
        Determine whether a verb token is grammatically negated.

        Parameters
        ----------
        verb_token : Any
            A spaCy ``Token`` representing the relation's verb.

        Returns
        -------
        bool
            True if a negation particle (e.g. "not", "n't") modifies
            this verb, False otherwise.
        """
        return any(child.dep_ == "neg" for child in verb_token.children)

    @staticmethod
    def _span_text(token: Any) -> str:
        """
        Expand a single token into its full noun-phrase text when possible.

        Using the token's dependency subtree (rather than the bare token)
        captures multi-word entities like "type 2 diabetes" instead of
        just "diabetes".

        Parameters
        ----------
        token : Any
            A spaCy ``Token`` anchoring a subject or object.

        Returns
        -------
        str
            The reconstructed, whitespace-normalized phrase text.
        """
        try:
            subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
            text = "".join(
                t.text_with_ws for t in subtree_tokens
            ).strip()
            return text if text else token.text
        except Exception:  # noqa: BLE001
            # Fall back to the bare token if subtree reconstruction fails.
            return token.text

    @staticmethod
    def _compute_confidence(
        is_prepositional: bool, is_passive: bool, is_negated: bool
    ) -> float:
        """
        Compute a heuristic confidence score for a relation.

        This is a transparent, explainable heuristic (not a learned
        model) intended as a reasonable default that can be swapped out
        later without changing this module's public interface.

        Parameters
        ----------
        is_prepositional : bool
            True if the object was reached through a preposition rather
            than being a direct object.
        is_passive : bool
            True if the subject was found via a passive-voice dependency.
        is_negated : bool
            True if the verb is grammatically negated.

        Returns
        -------
        float
            A confidence score clamped to the range [0.0, 1.0].
        """
        score = 0.75  # Baseline: both a subject and an object were found.

        if is_prepositional:
            score -= 0.15  # Prepositional links are weaker/more ambiguous.
        if is_passive:
            score -= 0.05  # Passive voice is slightly less direct.
        if is_negated:
            score -= 0.20  # A negated relation is less certain as stated.

        return max(0.0, min(1.0, round(score, 2)))

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_and_clean_text(text: Optional[str]) -> Optional[str]:
        """
        Validate and normalize raw input text before processing.

        Parameters
        ----------
        text : Optional[str]
            Raw candidate input.

        Returns
        -------
        Optional[str]
            The stripped text if it is a non-empty string, otherwise
            ``None`` (with a warning already logged).
        """
        if text is None:
            logger.warning("extract_relations received None; returning empty result.")
            return None
        if not isinstance(text, str):
            logger.warning(
                "extract_relations expected str, got %s; returning empty result.",
                type(text).__name__,
            )
            return None
        stripped = text.strip()
        if not stripped:
            logger.warning(
                "extract_relations received an empty/whitespace-only string; "
                "returning empty result."
            )
            return None
        return stripped

    def __repr__(self) -> str:
        """Return a concise, debug-friendly representation."""
        status = "loaded" if self.is_model_loaded() else "not loaded"
        return f"RelationExtractor(model_name='{self.model_name}', status='{status}')"


# ---------------------------------------------------------------------------
# Example usage (only runs when this file is executed directly).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_extractor = RelationExtractor(model_name="en_core_web_sm", strict=False)

    sample_text = (
        "Metformin reduces hepatic glucose production. "
        "Insulin resistance is caused by obesity."
    )

    print(f"Extractor status: {demo_extractor!r}")
    results = demo_extractor.extract_relations(sample_text)

    if results:
        print(f"Found {len(results)} relation(s):")
        for record in results:
            print(
                f"  - ({record['subject']!r}, {record['verb']!r}, "
                f"{record['object']!r}) confidence={record['confidence']}"
            )
    else:
        print(
            "No relations found (or model unavailable). "
            "Install a model with: python -m spacy download en_core_web_sm"
        )