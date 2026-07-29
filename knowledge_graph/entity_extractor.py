"""
entity_extractor.py
====================

Named Entity Recognition (NER) layer for the BioNexus knowledge graph
pipeline.

This module wraps spaCy / SciSpaCy NER models behind a single stable
interface (`EntityExtractor`) that:

    * Never raises an uncaught / built-in exception to the caller.
    * Never crashes on missing models, missing files, ``None`` input,
      or empty strings.
    * Returns structured dictionaries (never raw tuples) so downstream
      modules (relation_extractor, graph_builder) have a stable contract.
    * Supports custom entity labels via spaCy's ``EntityRuler``.

This file has **no dependency on any other module in this package** and
can be imported, instantiated, and tested completely on its own.

Example
-------
>>> from entity_extractor import EntityExtractor
>>> extractor = EntityExtractor(model_name="en_core_web_sm")
>>> entities = extractor.extract_entities(
...     "Aspirin inhibits COX-1 and reduces inflammation."
... )
>>> for ent in entities:
...     print(ent["text"], ent["label"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level logger.
#
# NOTE: This is a standard Python logging pattern, not a mutable global
# variable in the problematic sense -- no shared state is read or written
# here. Every class in this module still receives its own configuration
# and holds no hidden shared state between instances.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
#
# Defined locally (not imported from elsewhere) so this module remains
# fully self-contained and importable in isolation, per the "every module
# must work independently" requirement.
# ---------------------------------------------------------------------------
class ModelLoadError(Exception):
    """
    Raised when a spaCy / SciSpaCy language model cannot be loaded.

    This is raised instead of letting spaCy's raw ``OSError`` /
    ``ImportError`` propagate, so callers only ever need to catch one
    well-documented exception type from this module.
    """


class EntityExtractionError(Exception):
    """
    Raised when entity extraction fails for a reason unrelated to model
    loading (e.g. an unexpected internal spaCy failure on a specific
    document). Malformed input itself does NOT raise this -- it is
    handled gracefully and returns an empty result instead.
    """


class EntityExtractor:
    """
    High-level, fault-tolerant wrapper around spaCy / SciSpaCy NER models.

    The extractor is intentionally defensive: any failure that would
    normally raise ``ImportError``, ``OSError``, ``AttributeError``, or
    similar built-in exceptions is caught internally and converted into
    either a graceful fallback (empty result) or one of this module's
    custom, informative exceptions.

    Attributes
    ----------
    model_name : str
        Name of the spaCy / SciSpaCy model requested at construction time.
    nlp : Optional[Any]
        The loaded spaCy ``Language`` pipeline object, or ``None`` if
        loading failed and ``strict`` was set to ``False``.
    strict : bool
        If True, a failed model load raises ``ModelLoadError`` immediately.
        If False, the extractor stays alive in a "degraded" state where
        ``extract_entities`` will return an empty list and log an error,
        rather than crashing the whole pipeline.
    """

    # Known SciSpaCy model name prefixes, used only for informative
    # error messages / install hints -- not for any control-flow logic.
    _SCISPACY_PREFIXES = ("en_core_sci_", "en_ner_")

    def __init__(
        self,
        model_name: str = "en_core_sci_sm",
        custom_labels: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        strict: bool = True,
    ) -> None:
        """
        Initialize the extractor and attempt to load the requested model.

        Parameters
        ----------
        model_name : str, optional
            The spaCy or SciSpaCy model to load (default: "en_core_web_sm").
        custom_labels : Optional[Dict[str, List[Dict[str, Any]]]], optional
            A mapping suitable for spaCy's ``EntityRuler.add_patterns``.
            Keys are ignored (present for readability at the call site);
            values must be lists of spaCy pattern dicts, e.g.::

                {
                    "GENE_PATTERNS": [
                        {"label": "GENE", "pattern": "TP53"},
                        {"label": "GENE", "pattern": "BRCA1"},
                    ]
                }

        strict : bool, optional
            If True (default), failure to load the model raises
            ``ModelLoadError``. If False, the extractor is created in a
            degraded state instead of raising, and every subsequent call
            to ``extract_entities`` will simply return ``[]``.

        Raises
        ------
        ModelLoadError
            If ``strict`` is True and the model cannot be loaded (missing
            package, missing model data, or spaCy not installed at all).
        """
        self.model_name: str = model_name if isinstance(model_name, str) else ""
        self.strict: bool = bool(strict)
        self.nlp: Optional[Any] = None
        self._model_loaded: bool = False

        self._load_model()

        if custom_labels:
            try:
                self.add_custom_entity_labels(custom_labels)
            except Exception as exc:  # noqa: BLE001 - intentional broad guard
                # Custom labels are a "nice to have" enhancement; a failure
                # here should not prevent the extractor from functioning
                # with its base model.
                logger.warning(
                    "Failed to register custom entity labels for model "
                    "'%s': %s. Continuing with base model only.",
                    self.model_name,
                    exc,
                )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """
        Attempt to import spaCy and load ``self.model_name``.

        This method converts every possible failure mode (spaCy not
        installed, model not downloaded, corrupted model data) into
        either a raised ``ModelLoadError`` (strict mode) or a logged
        error with the extractor left in a safe, degraded state
        (non-strict mode).

        Raises
        ------
        ModelLoadError
            Only when ``self.strict`` is True.
        """
        if not self.model_name:
            message = (
                "No valid model_name was provided to EntityExtractor "
                "(received an empty string or non-string value)."
            )
            logger.error(message)
            if self.strict:
                raise ModelLoadError(message)
            return

        try:
            import spacy  # Imported lazily so this module can be imported
            # even in environments where spaCy is not installed yet --
            # e.g. for unit tests that only exercise error handling.
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
            self._model_loaded = True
            logger.info("Successfully loaded model '%s'.", self.model_name)
        except OSError as exc:
            # spaCy raises OSError when a model is not downloaded/present.
            hint = self._build_install_hint(self.model_name)
            message = (
                f"Model '{self.model_name}' could not be loaded. It is "
                f"likely not installed. {hint}"
            )
            logger.error("%s (original error: %s)", message, exc)
            if self.strict:
                raise ModelLoadError(message) from exc
            self.nlp = None
            self._model_loaded = False
        except Exception as exc:  # noqa: BLE001 - final safety net
            # Catches anything unexpected (corrupted model, version
            # mismatch, etc.) so a bad model never crashes the caller.
            message = f"Unexpected error while loading model '{self.model_name}'."
            logger.error("%s (original error: %s)", message, exc)
            if self.strict:
                raise ModelLoadError(message) from exc
            self.nlp = None
            self._model_loaded = False

    def _build_install_hint(self, model_name: str) -> str:
        """
        Build a human-readable installation hint for a given model name.

        Parameters
        ----------
        model_name : str
            The model that failed to load.

        Returns
        -------
        str
            A suggested shell command to install the model.
        """
        if model_name.startswith(self._SCISPACY_PREFIXES):
            return (
                "SciSpaCy models are distributed as direct package URLs. "
                "See https://allenai.github.io/scispacy/ for the correct "
                "pip install command for this model."
            )
        return f"Try:  python -m spacy download {model_name}"

    def is_model_loaded(self) -> bool:
        """
        Report whether a usable NLP pipeline is currently loaded.

        Returns
        -------
        bool
            True if a spaCy pipeline was successfully loaded and is ready
            to process text, False otherwise.
        """
        return self._model_loaded and self.nlp is not None

    # ------------------------------------------------------------------
    # Custom entity labels
    # ------------------------------------------------------------------
    def add_custom_entity_labels(
        self, custom_labels: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """
        Register custom entity patterns using spaCy's ``EntityRuler``.

        This allows BioNexus to recognize domain-specific entities (e.g.
        gene symbols, drug names) that a general-purpose or even a
        biomedical model may not label correctly out of the box.

        Parameters
        ----------
        custom_labels : Dict[str, List[Dict[str, Any]]]
            A mapping from an arbitrary group name to a list of spaCy
            pattern dictionaries (each must contain at least "label"
            and "pattern" keys, per spaCy's EntityRuler format).

        Raises
        ------
        EntityExtractionError
            If the pipeline is not loaded, or if the patterns are
            malformed and spaCy rejects them.
        """
        if not self.is_model_loaded():
            raise EntityExtractionError(
                "Cannot add custom entity labels: no model is currently "
                "loaded on this EntityExtractor instance."
            )

        if not isinstance(custom_labels, dict):
            logger.warning(
                "add_custom_entity_labels expected a dict, got %s. "
                "Ignoring custom labels.",
                type(custom_labels).__name__,
            )
            return

        # Flatten all pattern lists into one, skipping anything malformed
        # rather than letting one bad entry abort the whole registration.
        all_patterns: List[Dict[str, Any]] = []
        for group_name, patterns in custom_labels.items():
            if not isinstance(patterns, list):
                logger.warning(
                    "Custom label group '%s' is not a list; skipping.",
                    group_name,
                )
                continue
            for pattern in patterns:
                if isinstance(pattern, dict) and "label" in pattern and "pattern" in pattern:
                    all_patterns.append(pattern)
                else:
                    logger.warning(
                        "Skipping malformed pattern in group '%s': %r",
                        group_name,
                        pattern,
                    )

        if not all_patterns:
            logger.info("No valid custom entity patterns were provided.")
            return

        try:
            if "entity_ruler" in self.nlp.pipe_names:
                ruler = self.nlp.get_pipe("entity_ruler")
            else:
                # Place the ruler before the statistical NER component so
                # its explicit matches take priority, when possible.
                if "ner" in self.nlp.pipe_names:
                    ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                else:
                    ruler = self.nlp.add_pipe("entity_ruler")
            ruler.add_patterns(all_patterns)
            logger.info(
                "Registered %d custom entity pattern(s).", len(all_patterns)
            )
        except Exception as exc:  # noqa: BLE001
            raise EntityExtractionError(
                f"Failed to register custom entity patterns: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_entities(self, text: Optional[str]) -> List[Dict[str, Any]]:
        """
        Extract named entities from a piece of text.

        This method is designed to never raise on bad input. ``None``,
        non-string values, empty strings, and whitespace-only strings
        all result in an empty list plus a logged warning -- they do
        NOT raise ``TypeError`` or ``ValueError``.

        Parameters
        ----------
        text : Optional[str]
            The input text to analyze.

        Returns
        -------
        List[Dict[str, Any]]
            A list of structured entity dictionaries, each shaped as::

                {
                    "text": str,          # surface form of the entity
                    "label": str,         # entity type / label
                    "start_char": int,    # character offset (inclusive)
                    "end_char": int,      # character offset (exclusive)
                    "metadata": {
                        "model": str,     # model used for extraction
                        "lemma": str,     # lemmatized form, if available
                    },
                }

            Returns ``[]`` (never ``None``) if extraction could not be
            performed for any reason.

        Raises
        ------
        EntityExtractionError
            Only for genuine internal failures during processing of
            otherwise valid, non-empty text (e.g. an unexpected spaCy
            runtime error). Bad input never triggers this.
        """
        cleaned_text = self._validate_and_clean_text(text)
        if cleaned_text is None:
            # Already logged inside _validate_and_clean_text.
            return []

        if not self.is_model_loaded():
            logger.error(
                "extract_entities called but no model is loaded "
                "(model_name='%s'). Returning empty result.",
                self.model_name,
            )
            return []

        try:
            doc = self.nlp(cleaned_text)
        except Exception as exc:  # noqa: BLE001
            raise EntityExtractionError(
                f"spaCy pipeline failed while processing text: {exc}"
            ) from exc

        return self._build_entity_records(doc)

    def _build_entity_records(self, doc: Any) -> List[Dict[str, Any]]:
        """
        Convert a spaCy ``Doc`` object's entities into structured dicts.

        Parameters
        ----------
        doc : Any
            A processed spaCy ``Doc`` object.

        Returns
        -------
        List[Dict[str, Any]]
            Structured entity records. Any single malformed entity span
            is skipped (and logged) rather than aborting the whole batch.
        """
        records: List[Dict[str, Any]] = []
        try:
            entities = list(doc.ents)
        except AttributeError as exc:
            logger.error(
                "The processed document has no 'ents' attribute; the "
                "loaded pipeline may not include an NER component. "
                "Error: %s",
                exc,
            )
            return records

        for ent in entities:
            try:
                records.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "metadata": {
                            "model": self.model_name,
                            "lemma": ent.lemma_ if hasattr(ent, "lemma_") else "",
                        },
                    }
                )
            except AttributeError as exc:
                logger.warning("Skipping malformed entity span: %s", exc)
                continue

        return records

    # ------------------------------------------------------------------
    # Helpers
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
            logger.warning("extract_entities received None; returning empty result.")
            return None
        if not isinstance(text, str):
            logger.warning(
                "extract_entities expected str, got %s; returning empty result.",
                type(text).__name__,
            )
            return None
        stripped = text.strip()
        if not stripped:
            logger.warning(
                "extract_entities received an empty/whitespace-only string; "
                "returning empty result."
            )
            return None
        return stripped

    def get_supported_labels(self) -> List[str]:
        """
        List the entity labels the currently loaded pipeline can produce.

        Returns
        -------
        List[str]
            Sorted list of entity label strings, or ``[]`` if no model
            is loaded or the pipeline has no NER-capable component.
        """
        if not self.is_model_loaded():
            return []
        try:
            if "ner" in self.nlp.pipe_names:
                return sorted(self.nlp.get_pipe("ner").labels)
            # Fall back to whatever pipe defines labels (e.g. entity_ruler).
            labels: set = set()
            for pipe_name in self.nlp.pipe_names:
                pipe = self.nlp.get_pipe(pipe_name)
                if hasattr(pipe, "labels"):
                    labels.update(pipe.labels)
            return sorted(labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not retrieve supported labels: %s", exc)
            return []

    def __repr__(self) -> str:
        """Return a concise, debug-friendly representation."""
        status = "loaded" if self.is_model_loaded() else "not loaded"
        return f"EntityExtractor(model_name='{self.model_name}', status='{status}')"


# ---------------------------------------------------------------------------
# Example usage (only runs when this file is executed directly, e.g.
# `python entity_extractor.py`). This never runs on import, so it cannot
# interfere with the package's public API.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Non-strict mode is used here purely for demonstration purposes, so
    # this example still runs even on machines without any spaCy model
    # installed -- it will simply print a warning and an empty list.
    demo_extractor = EntityExtractor(model_name="en_core_web_sm", strict=False)

    sample_text = (
        "Metformin is commonly used to treat type 2 diabetes by reducing "
        "hepatic glucose production."
    )

    print(f"Extractor status: {demo_extractor!r}")
    results = demo_extractor.extract_entities(sample_text)

    if results:
        print(f"Found {len(results)} entities:")
        for record in results:
            print(f"  - {record['text']!r} -> {record['label']}")
    else:
        print(
            "No entities found (or model unavailable). "
            "Install a model with: python -m spacy download en_core_web_sm"
        )