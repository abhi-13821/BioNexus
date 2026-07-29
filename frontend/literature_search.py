"""
frontend/literature_search.py
==============================================================================
BioNexus — AI-Powered Multi-Source Biomedical Literature Search
==============================================================================

A single self-contained Streamlit module that aggregates biomedical literature
from PubMed, Europe PMC, CrossRef, Semantic Scholar, OpenAlex, DOAJ, bioRxiv
and medRxiv; deduplicates and ranks the merged results; enriches them with
lightweight AI-style summarization / entity extraction; renders interactive
Plotly analytics; and supports exporting results (CSV / JSON / TXT / BibTeX /
RIS / PDF) plus an auto-generated literature review.

Usage
-----
    # app.py
    from frontend.literature_search import show
    show()

No modification to the host app is required — everything (state, caching,
styling) is self-contained within this module.

Dependencies (all free / open source)
--------------------------------------
    streamlit, requests, pandas, plotly
Optional (graceful degradation if missing)
    fpdf2   -> PDF report export

Author: BioNexus Engineering
==============================================================================
"""

from __future__ import annotations


import re
import io
import csv
import json
import time
import html
import hashlib
import logging
import difflib
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET


import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

logger = logging.getLogger("bionexus.literature_search")
logging.basicConfig(level=logging.INFO)

try:
    from fpdf import FPDF, XPos, YPos  # fpdf2
    _HAS_FPDF = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_FPDF = False


# ==============================================================================
# 0B. EMBEDDINGS INTEGRATION (BioNexus semantic search)
# ==============================================================================
# Reuses the existing, already-tested `embeddings` package as-is: no changes
# to embedding_generator.py, embedding_store.py, embedding_search.py,
# embedding_manager.py, cache.py, models.py, or utils.py are made or needed.
from embeddings.embedding_generator import SentenceTransformerEmbeddingGenerator
from embeddings.embedding_store import JSONEmbeddingStore
from embeddings.embedding_manager import EmbeddingManager, EmbeddingManagerError
from embeddings.models import EmbeddingMetadata

# Where literature-search embeddings are persisted (separate from any other
# embeddings the platform may build, e.g. knowledge-graph-derived text).
_EMBEDDINGS_STORAGE_PATH = "data/embeddings/literature_search_embeddings.json"
# Distinguishes literature-search embeddings from other source_types that
# may later share the same underlying embeddings store.
_EMBEDDING_SOURCE_TYPE = "literature_paper"
_DEFAULT_SEMANTIC_TOP_K = 10


@st.cache_resource(show_spinner=False)
def _get_embedding_manager() -> EmbeddingManager:
    """
    Build (once per Streamlit server process) the EmbeddingManager used to
    turn literature search results into vector embeddings for semantic
    search.

    Wrapped in ``st.cache_resource`` so the underlying sentence-transformers
    model, the on-disk embedding store, and the in-memory cache are all
    constructed exactly once per server process rather than on every
    Streamlit rerun — model loading is by far the most expensive step in
    this pipeline, and re-loading it per rerun would make the search page
    unusably slow.

    Returns
    -------
    EmbeddingManager
        A manager wired to the default
        ``sentence-transformers/all-MiniLM-L6-v2`` generator and a
        JSON-backed store dedicated to literature search embeddings.
    """
    logger.info(
        "Initializing EmbeddingManager for literature search (storage_path='%s').",
        _EMBEDDINGS_STORAGE_PATH,
    )
    generator = SentenceTransformerEmbeddingGenerator()
    store = JSONEmbeddingStore(storage_path=_EMBEDDINGS_STORAGE_PATH)
    return EmbeddingManager(generator=generator, store=store)


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration for the literature search module."""

    APP_NAME = "BioNexus Literature Intelligence"
    REQUEST_TIMEOUT = 12          # seconds, per outbound HTTP call
    MAX_WORKERS = 8               # thread pool size for concurrent source calls
    DEFAULT_MAX_RESULTS = 40      # per-source cap before merging
    MIN_QUERY_LENGTH = 3          # configurable minimum query length
    CACHE_TTL_SECONDS = 60 * 30   # 30 minutes
    USER_AGENT = "BioNexus-LiteratureSearch/1.0 (mailto:research@bionexus.app)"

    # --- Source endpoints -----------------------------------------------
    PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    CROSSREF_WORKS = "https://api.crossref.org/works"

    SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
    SEMANTIC_SCHOLAR_FIELDS = (
        "title,abstract,authors,year,venue,externalIds,citationCount,"
        "influentialCitationCount,referenceCount,isOpenAccess,openAccessPdf,"
        "publicationTypes,fieldsOfStudy,url"
    )

    OPENALEX_WORKS = "https://api.openalex.org/works"

    DOAJ_SEARCH = "https://doaj.org/api/search/articles"

    BIORXIV_DETAILS = "https://api.biorxiv.org/details/biorxiv"
    MEDRXIV_DETAILS = "https://api.biorxiv.org/details/medrxiv"

    SOURCES = [
        "PubMed", "PubMed Central", "Europe PMC", "CrossRef",
        "Semantic Scholar", "OpenAlex", "DOAJ", "bioRxiv", "medRxiv",
    ]


# ==============================================================================
# 2. DOMAIN MODEL
# ==============================================================================

@dataclass
class Paper:
    """Normalized representation of a single research paper, regardless of
    which upstream source it came from."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    citation_count: Optional[int] = None
    influential_citations: Optional[int] = None
    reference_count: Optional[int] = None
    keywords: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    abstract: str = ""
    publisher: str = ""
    publication_type: str = ""
    is_open_access: bool = False
    pdf_url: str = ""
    url: str = ""
    source: str = ""
    relevance_score: float = 0.0

    # -- derived / AI-enriched fields, filled in lazily -------------------
    short_summary: str = ""
    detailed_summary: str = ""
    entities: dict[str, list[str]] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Stable identifier used for deduplication and Streamlit keys."""
        basis = self.doi or self.pmid or self.pmcid or self._title_key()
        return hashlib.md5(basis.encode("utf-8", "ignore")).hexdigest()

    def _title_key(self) -> str:
        return re.sub(r"[^a-z0-9]+", "", self.title.lower())[:120]

    def author_str(self, limit: int = 3) -> str:
        if not self.authors:
            return "Unknown authors"
        shown = self.authors[:limit]
        suffix = f" et al. ({len(self.authors)} total)" if len(self.authors) > limit else ""
        return ", ".join(shown) + suffix

    def to_flat_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = "; ".join(str(x) for x in v)
            elif isinstance(v, dict):
                d[k] = json.dumps(v)
        return d


@dataclass
class SearchFilters:
    """User-selected search constraints."""

    year_from: Optional[int] = None
    year_to: Optional[int] = None
    journal: str = ""
    author: str = ""
    article_type: str = "Any"          # Any/Review/Clinical Trial/Meta-analysis
    open_access_only: bool = False
    human_studies_only: bool = False
    animal_studies_only: bool = False
    free_full_text_only: bool = False
    language: str = "Any"
    max_results: int = Config.DEFAULT_MAX_RESULTS
    sort_by: str = "Most Relevant"     # Newest/Oldest/Most Relevant/Highest Citation Count


# ==============================================================================
# 3. QUERY VALIDATION
# ==============================================================================

_BIOMEDICAL_HINTS = {
    "cancer", "tumor", "tumour", "gene", "genome", "protein", "cell", "clinical",
    "disease", "syndrome", "therapy", "treatment", "drug", "receptor", "enzyme",
    "diabetes", "cardio", "neuro", "immune", "immunology", "oncology", "virus",
    "bacteria", "infection", "vaccine", "mutation", "dna", "rna", "biomarker",
    "pathway", "inflammation", "chronic", "acute", "diagnosis", "prognosis",
    "trial", "patient", "disorder", "metabolic", "molecular", "pharma",
    "antibody", "antigen", "pathogen", "epidemiology", "genetic", "cellular",
    "tissue", "organ", "surgery", "screening", "biopsy", "assay", "cohort",
    "randomized", "meta-analysis", "systematic review", "risk factor",
}

_KNOWN_GENE_DRUG_PATTERN = re.compile(r"^[A-Z0-9]{2,10}$")


class QueryValidationResult:
    def __init__(self, is_valid: bool, message: str = "", looks_biomedical: bool = True):
        self.is_valid = is_valid
        self.message = message
        self.looks_biomedical = looks_biomedical


def validate_query(query: str, min_length: int = Config.MIN_QUERY_LENGTH) -> QueryValidationResult:
    """Smart query validation with friendly error messages.

    Rejects empty, whitespace-only, purely numeric, purely symbolic, single
    character (unless a recognized gene/drug-style acronym), and generally
    meaningless queries. Also flags (without blocking) queries that do not
    appear biomedical in nature, so the UI can show a soft warning.
    """
    if query is None:
        return QueryValidationResult(False, "Please enter a search term.")

    trimmed = query.strip()

    if not trimmed:
        return QueryValidationResult(False, "Search query cannot be empty. Try a term like 'EGFR' or 'lung cancer'.")

    if len(trimmed) < min_length and not _KNOWN_GENE_DRUG_PATTERN.match(trimmed):
        return QueryValidationResult(
            False,
            f"Your query is too short (minimum {min_length} characters). "
            f"Try something more specific, e.g. 'BRCA1' or 'aspirin'.",
        )

    if trimmed.isdigit():
        return QueryValidationResult(False, "A purely numeric query won't return meaningful results. "
                                             "Try a disease name, gene symbol, or drug name instead.")

    if not re.search(r"[a-zA-Z]", trimmed):
        return QueryValidationResult(False, "Your query needs to contain at least one letter or word.")

    # Reject strings of pure repeated/random symbols or gibberish
    letters_only = re.sub(r"[^a-zA-Z]", "", trimmed)
    if len(letters_only) < 2:
        return QueryValidationResult(False, "This doesn't look like a valid search term. "
                                             "Please try a biomedical keyword or phrase.")

    # crude gibberish heuristic: no vowels in a long alphabetic token
    if len(letters_only) >= 6 and not re.search(r"[aeiouAEIOU]", letters_only):
        return QueryValidationResult(False, "This query looks like it might be a typo. "
                                             "Please double check the spelling.")

    looks_biomedical = _looks_biomedical(trimmed)
    return QueryValidationResult(True, "", looks_biomedical)


def _looks_biomedical(query: str) -> bool:
    lower = query.lower()
    if any(hint in lower for hint in _BIOMEDICAL_HINTS):
        return True
    # short all-caps tokens are frequently gene/protein symbols (EGFR, BRCA1, TP53)
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if any(re.match(r"^[A-Z0-9]{2,8}$", t) for t in tokens):
        return True
    # otherwise unknown — not necessarily wrong, just unverified
    return len(tokens) <= 6  # short queries get the benefit of the doubt


def parse_boolean_query(query: str) -> dict[str, Any]:
    """Parse simple boolean syntax (AND / OR / NOT / "quoted phrases") into a
    structure usable for building source-specific query strings."""
    quoted = re.findall(r'"([^"]+)"', query)
    remainder = re.sub(r'"[^"]+"', "", query)
    tokens = remainder.split()

    must, should, must_not = [], [], []
    mode = "AND"
    for tok in tokens:
        upper = tok.upper()
        if upper in ("AND", "OR", "NOT"):
            mode = upper
            continue
        target = {"AND": must, "OR": should, "NOT": must_not}[mode]
        target.append(tok)

    return {
        "phrases": quoted,
        "must": must,
        "should": should,
        "must_not": must_not,
        "raw": query,
    }


def build_plain_query(parsed: dict[str, Any]) -> str:
    """Recombine a parsed boolean structure into a plain-text query string
    suitable for APIs that don't support native boolean syntax."""
    parts = list(parsed["phrases"]) + parsed["must"] + parsed["should"]
    return " ".join(parts) if parts else parsed["raw"]


# ==============================================================================
# 4. HTTP HELPERS
# ==============================================================================

class SourceUnavailableError(Exception):
    """Raised (and always caught) when an upstream source cannot be reached."""


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    headers = {"User-Agent": Config.USER_AGENT, **(headers or {})}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=Config.REQUEST_TIMEOUT)
        if resp.status_code == 429:
            raise SourceUnavailableError("Rate limited")
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout as exc:
        raise SourceUnavailableError(f"Timeout contacting {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise SourceUnavailableError(f"No internet / connection error reaching {url}") from exc
    except requests.exceptions.HTTPError as exc:
        raise SourceUnavailableError(f"HTTP error from {url}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise SourceUnavailableError(f"Request failed for {url}: {exc}") from exc


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ==============================================================================
# 5. SOURCE CONNECTORS
# ==============================================================================
# Each connector implements .search(query, filters) -> list[Paper] and never
# raises — failures are logged and an empty list is returned so the
# aggregator can gracefully fall back to the remaining sources.

class BaseConnector:
    name = "Base"

    def search(self, query: str, filters: SearchFilters) -> list[Paper]:
        try:
            return self._search(query, filters)
        except SourceUnavailableError as exc:
            logger.warning("%s unavailable: %s", self.name, exc)
            return []
        except Exception as exc:  # noqa: BLE001 - defensive catch-all per spec
            logger.warning("%s failed unexpectedly: %s", self.name, exc)
            return []

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        raise NotImplementedError


class PubMedConnector(BaseConnector):
    name = "PubMed"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        term = query
        if filters.year_from or filters.year_to:
            lo = filters.year_from or 1900
            hi = filters.year_to or dt.date.today().year
            term += f" AND ({lo}:{hi}[dp])"
        if filters.article_type and filters.article_type != "Any":
            term += f" AND {filters.article_type}[pt]"

        params = {
            "db": "pubmed", "term": term, "retmode": "json",
            "retmax": filters.max_results, "sort": self._sort_param(filters.sort_by),
        }
        resp = _get(Config.PUBMED_ESEARCH, params=params)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        return self._fetch_details(ids)

    @staticmethod
    def _sort_param(sort_by: str) -> str:
        return {"Newest": "pub+date", "Most Relevant": "relevance"}.get(sort_by, "relevance")

    def _fetch_details(self, ids: list[str]) -> list[Paper]:
        resp = _get(Config.PUBMED_EFETCH, params={
            "db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml",
        })
        papers: list[Paper] = []
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return []
        for article in root.findall(".//PubmedArticle"):
            papers.append(self._parse_article(article))
        return papers

    def _parse_article(self, article: ET.Element) -> Paper:
        def text(el, default=""):
            return el.text.strip() if el is not None and el.text else default

        title = text(article.find(".//ArticleTitle"))
        abstract_parts = [text(a) for a in article.findall(".//Abstract/AbstractText")]
        abstract = " ".join(p for p in abstract_parts if p)

        authors = []
        affiliations = []
        for author in article.findall(".//AuthorList/Author"):
            last = text(author.find("LastName"))
            fore = text(author.find("ForeName"))
            if last:
                authors.append(f"{fore} {last}".strip())
            aff = author.find(".//AffiliationInfo/Affiliation")
            if aff is not None and aff.text:
                affiliations.append(aff.text.strip())

        journal = text(article.find(".//Journal/Title"))
        year_text = text(article.find(".//JournalIssue/PubDate/Year"))
        volume = text(article.find(".//JournalIssue/Volume"))
        issue = text(article.find(".//JournalIssue/Issue"))
        pages = text(article.find(".//Pagination/MedlinePgn"))

        pmid = text(article.find(".//PMID"))
        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = text(aid)
            elif aid.get("IdType") == "pmc":
                pass

        pmcid = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "pmc":
                pmcid = text(aid)

        mesh_terms = [text(m) for m in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")]
        pub_types = [text(p) for p in article.findall(".//PublicationTypeList/PublicationType")]

        return Paper(
            title=title or "(untitled)",
            authors=authors,
            affiliations=list(dict.fromkeys(affiliations)),
            journal=journal,
            year=_safe_int(year_text),
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            mesh_terms=mesh_terms,
            abstract=abstract,
            publication_type=", ".join(pub_types),
            is_open_access=bool(pmcid),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            pdf_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
            source="PubMed",
        )


class PMCConnector(BaseConnector):
    """PubMed Central — searched via the same E-utilities, restricted to the
    PMC (full-text, open access) database."""
    name = "PubMed Central"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        params = {"db": "pmc", "term": query, "retmode": "json", "retmax": filters.max_results}
        resp = _get(Config.PUBMED_ESEARCH, params=params)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        papers = []
        for pmcid in ids:
            papers.append(Paper(
                title=f"PMC Article {pmcid}",
                pmcid=pmcid,
                is_open_access=True,
                url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                source="PubMed Central",
            ))
        return papers


class EuropePMCConnector(BaseConnector):
    name = "Europe PMC"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        params = {
            "query": query, "format": "json", "pageSize": filters.max_results,
            "resultType": "core",
        }
        resp = _get(Config.EUROPEPMC_SEARCH, params=params)
        results = resp.json().get("resultList", {}).get("result", [])
        return [self._parse(r) for r in results]

    def _parse(self, r: dict) -> Paper:
        authors = [a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()]
        return Paper(
            title=r.get("title", "").strip() or "(untitled)",
            authors=authors,
            journal=r.get("journalTitle", ""),
            year=_safe_int(r.get("pubYear")),
            doi=r.get("doi", ""),
            pmid=r.get("pmid", ""),
            pmcid=r.get("pmcid", ""),
            citation_count=_safe_int(r.get("citedByCount")),
            abstract=r.get("abstractText", ""),
            publisher=r.get("publisher", ""),
            is_open_access=(r.get("isOpenAccess") == "Y"),
            url=f"https://europepmc.org/article/{r.get('source','MED')}/{r.get('id','')}",
            source="Europe PMC",
        )


class CrossRefConnector(BaseConnector):
    name = "CrossRef"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        params = {"query": query, "rows": filters.max_results}
        if filters.year_from:
            params["filter"] = f"from-pub-date:{filters.year_from}-01-01"
        resp = _get(Config.CROSSREF_WORKS, params=params)
        items = resp.json().get("message", {}).get("items", [])
        return [self._parse(i) for i in items]

    def _parse(self, item: dict) -> Paper:
        title = " ".join(item.get("title", []) or ["(untitled)"])
        authors = []
        for a in item.get("author", []) or []:
            name = " ".join(filter(None, [a.get("given"), a.get("family")]))
            if name:
                authors.append(name)
        year = None
        date_parts = (item.get("published-print") or item.get("published-online") or {}).get("date-parts")
        if date_parts and date_parts[0]:
            year = _safe_int(date_parts[0][0])
        return Paper(
            title=title,
            authors=authors,
            journal=" ".join(item.get("container-title", []) or []),
            year=year,
            volume=item.get("volume", ""),
            issue=item.get("issue", ""),
            pages=item.get("page", ""),
            doi=item.get("DOI", ""),
            citation_count=_safe_int(item.get("is-referenced-by-count")),
            publisher=item.get("publisher", ""),
            publication_type=item.get("type", ""),
            url=item.get("URL", ""),
            source="CrossRef",
        )


class SemanticScholarConnector(BaseConnector):
    name = "Semantic Scholar"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        params = {"query": query, "limit": min(filters.max_results, 100),
                   "fields": Config.SEMANTIC_SCHOLAR_FIELDS}
        resp = _get(Config.SEMANTIC_SCHOLAR_SEARCH, params=params)
        data = resp.json().get("data", [])
        return [self._parse(d) for d in data]

    def _parse(self, d: dict) -> Paper:
        ext = d.get("externalIds") or {}
        oa_pdf = (d.get("openAccessPdf") or {}).get("url", "")
        return Paper(
            title=d.get("title", "") or "(untitled)",
            authors=[a.get("name", "") for a in d.get("authors", []) or []],
            journal=d.get("venue", ""),
            year=_safe_int(d.get("year")),
            doi=ext.get("DOI", ""),
            pmid=ext.get("PubMed", ""),
            citation_count=_safe_int(d.get("citationCount")),
            influential_citations=_safe_int(d.get("influentialCitationCount")),
            reference_count=_safe_int(d.get("referenceCount")),
            abstract=d.get("abstract", "") or "",
            publication_type=", ".join(d.get("publicationTypes") or []),
            keywords=d.get("fieldsOfStudy") or [],
            is_open_access=bool(d.get("isOpenAccess")),
            pdf_url=oa_pdf,
            url=d.get("url", ""),
            source="Semantic Scholar",
        )


class OpenAlexConnector(BaseConnector):
    name = "OpenAlex"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        params = {"search": query, "per-page": min(filters.max_results, 50)}
        resp = _get(Config.OPENALEX_WORKS, params=params)
        results = resp.json().get("results", [])
        return [self._parse(r) for r in results]

    def _parse(self, r: dict) -> Paper:
        authors = [a.get("author", {}).get("display_name", "")
                   for a in r.get("authorships", []) or []]
        primary = r.get("primary_location") or {}
        source_info = primary.get("source") or {}
        return Paper(
            title=r.get("title", "") or "(untitled)",
            authors=[a for a in authors if a],
            journal=source_info.get("display_name", ""),
            year=_safe_int(r.get("publication_year")),
            doi=(r.get("doi") or "").replace("https://doi.org/", ""),
            citation_count=_safe_int(r.get("cited_by_count")),
            publisher=source_info.get("host_organization_name", "") or "",
            publication_type=r.get("type", ""),
            is_open_access=bool((r.get("open_access") or {}).get("is_oa")),
            pdf_url=primary.get("pdf_url", "") or "",
            url=r.get("id", ""),
            keywords=[c.get("display_name", "") for c in r.get("concepts", []) or []][:8],
            source="OpenAlex",
        )


class DOAJConnector(BaseConnector):
    name = "DOAJ"

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        url = f"{Config.DOAJ_SEARCH}/{requests.utils.quote(query)}"
        resp = _get(url, params={"pageSize": filters.max_results})
        results = resp.json().get("results", [])
        return [self._parse(r) for r in results]

    def _parse(self, r: dict) -> Paper:
        bib = r.get("bibjson", {})
        authors = [a.get("name", "") for a in bib.get("author", []) or []]
        journal = (bib.get("journal") or {}).get("title", "")
        year = _safe_int(bib.get("year"))
        doi = ""
        for ident in bib.get("identifier", []) or []:
            if ident.get("type") == "doi":
                doi = ident.get("id", "")
        link = ""
        for lk in bib.get("link", []) or []:
            if lk.get("type") == "fulltext":
                link = lk.get("url", "")
        return Paper(
            title=bib.get("title", "") or "(untitled)",
            authors=authors,
            journal=journal,
            year=year,
            doi=doi,
            abstract=bib.get("abstract", "") or "",
            keywords=bib.get("keywords", []) or [],
            is_open_access=True,
            url=link,
            source="DOAJ",
        )


class PreprintConnector(BaseConnector):
    """Shared implementation for bioRxiv / medRxiv, whose public API is
    date-range based rather than full-text search. We pull recent preprints
    and filter client-side by keyword match against title/abstract, noting
    this as a known limitation of the free public API."""

    name = "bioRxiv"
    base_url = Config.BIORXIV_DETAILS
    lookback_days = 180

    def _search(self, query: str, filters: SearchFilters) -> list[Paper]:
        today = dt.date.today()
        start = today - dt.timedelta(days=self.lookback_days)
        url = f"{self.base_url}/{start.isoformat()}/{today.isoformat()}/0"
        resp = _get(url)
        collection = resp.json().get("collection", [])
        query_terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query)]

        matched = []
        for item in collection:
            haystack = f"{item.get('title','')} {item.get('abstract','')}".lower()
            if any(term in haystack for term in query_terms):
                matched.append(self._parse(item))
            if len(matched) >= filters.max_results:
                break
        return matched

    def _parse(self, item: dict) -> Paper:
        authors = [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()]
        doi = item.get("doi", "")
        return Paper(
            title=item.get("title", "") or "(untitled)",
            authors=authors,
            journal=self.name,
            year=_safe_int((item.get("date") or "0000")[:4]),
            doi=doi,
            abstract=item.get("abstract", "") or "",
            is_open_access=True,
            url=f"https://doi.org/{doi}" if doi else "",
            publication_type="Preprint",
            source=self.name,
        )


class MedRxivConnector(PreprintConnector):
    name = "medRxiv"
    base_url = Config.MEDRXIV_DETAILS


# Registry of all connectors, instantiated once.
CONNECTORS: list[BaseConnector] = [
    PubMedConnector(), PMCConnector(), EuropePMCConnector(), CrossRefConnector(),
    SemanticScholarConnector(), OpenAlexConnector(), DOAJConnector(),
    PreprintConnector(), MedRxivConnector(),
]


# ==============================================================================
# 6. AGGREGATION: SEARCH, DEDUPE, RANK
# ==============================================================================

def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    """Merge duplicate papers across sources using DOI/PMID exact match first,
    falling back to fuzzy title similarity. Keeps the most information-rich
    record, backfilling missing fields from duplicates."""
    by_key: dict[str, Paper] = {}
    unresolved: list[Paper] = []

    for p in papers:
        key = p.doi.lower() if p.doi else (f"pmid:{p.pmid}" if p.pmid else "")
        if key:
            if key in by_key:
                by_key[key] = _merge_paper(by_key[key], p)
            else:
                by_key[key] = p
        else:
            unresolved.append(p)

    merged = list(by_key.values())

    # Fuzzy-match unresolved (no DOI/PMID) records against the merged pool.
    for p in unresolved:
        match = None
        for existing in merged:
            if existing.title and p.title and _title_similarity(existing.title, p.title) > 0.9:
                match = existing
                break
        if match:
            merged[merged.index(match)] = _merge_paper(match, p)
        else:
            merged.append(p)

    return merged


def _merge_paper(primary: Paper, secondary: Paper) -> Paper:
    """Fill gaps in `primary` using non-empty fields from `secondary`."""
    for f in primary.__dataclass_fields__:
        pv, sv = getattr(primary, f), getattr(secondary, f)
        if pv in (None, "", [], {}) and sv not in (None, "", [], {}):
            setattr(primary, f, sv)
    if secondary.source and secondary.source not in primary.source:
        primary.source = f"{primary.source}, {secondary.source}" if primary.source else secondary.source
    if secondary.citation_count and (primary.citation_count or 0) < secondary.citation_count:
        primary.citation_count = secondary.citation_count
    return primary


def apply_filters(papers: list[Paper], filters: SearchFilters) -> list[Paper]:
    out = papers
    if filters.year_from:
        out = [p for p in out if not p.year or p.year >= filters.year_from]
    if filters.year_to:
        out = [p for p in out if not p.year or p.year <= filters.year_to]
    if filters.journal:
        j = filters.journal.lower()
        out = [p for p in out if j in (p.journal or "").lower()]
    if filters.author:
        a = filters.author.lower()
        out = [p for p in out if any(a in au.lower() for au in p.authors)]
    if filters.open_access_only:
        out = [p for p in out if p.is_open_access]
    if filters.free_full_text_only:
        out = [p for p in out if p.pdf_url or p.pmcid]
    if filters.article_type and filters.article_type != "Any":
        t = filters.article_type.lower()
        out = [p for p in out if t in (p.publication_type or "").lower()]
    if filters.human_studies_only:
        out = [p for p in out if "animal" not in (p.abstract or "").lower()[:400]]
    if filters.animal_studies_only:
        out = [p for p in out if any(term in (p.abstract or "").lower()
                                      for term in ("mouse", "mice", "rat ", "murine", "animal model"))]
    return out


def score_relevance(paper: Paper, query_terms: list[str]) -> float:
    """Lightweight TF-based relevance score across title/abstract/keywords,
    with bonuses for citation count and recency."""
    text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
    term_hits = sum(text.count(t.lower()) for t in query_terms if t)
    title_hits = sum(paper.title.lower().count(t.lower()) for t in query_terms if t)

    score = term_hits * 1.0 + title_hits * 3.0
    if paper.citation_count:
        score += min(paper.citation_count, 500) * 0.02
    if paper.year:
        age = max(dt.date.today().year - paper.year, 0)
        score += max(5 - age, 0) * 0.5
    return round(score, 3)


def rank_papers(papers: list[Paper], query: str, sort_by: str) -> list[Paper]:
    terms = re.findall(r"[a-zA-Z0-9]+", query)
    for p in papers:
        p.relevance_score = score_relevance(p, terms)

    if sort_by == "Newest":
        return sorted(papers, key=lambda p: (p.year or 0), reverse=True)
    if sort_by == "Oldest":
        return sorted(papers, key=lambda p: (p.year or 9999))
    if sort_by == "Highest Citation Count":
        return sorted(papers, key=lambda p: (p.citation_count or 0), reverse=True)
    return sorted(papers, key=lambda p: p.relevance_score, reverse=True)


def run_federated_search(query: str, filters: SearchFilters,
                          sources: list[str]) -> tuple[list[Paper], dict[str, str]]:
    """Query all enabled sources concurrently, merge, dedupe, filter and rank.

    Returns the ranked paper list plus a status dict of {source: status_msg}
    for transparent error reporting in the UI.
    """
    active = [c for c in CONNECTORS if c.name in sources]
    results: list[Paper] = []
    status: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as pool:
        future_map = {pool.submit(c.search, query, filters): c for c in active}
        for future in as_completed(future_map):
            connector = future_map[future]
            try:
                papers = future.result()
                status[connector.name] = f"OK ({len(papers)} results)" if papers else "No results"
                results.extend(papers)
            except Exception as exc:  # noqa: BLE001
                status[connector.name] = f"Failed: {exc}"

    merged = dedupe_papers(results)
    filtered = apply_filters(merged, filters)
    ranked = rank_papers(filtered, query, filters.sort_by)
    return ranked[: filters.max_results * 2], status


# ==============================================================================
# 6B. EMBEDDINGS INTEGRATION — paper <-> EmbeddingMetadata + ingestion
# ==============================================================================

def _paper_to_embedding_metadata(paper: Paper) -> Optional[EmbeddingMetadata]:
    """
    Convert a `Paper` into `EmbeddingMetadata` suitable for the embeddings
    package, or return ``None`` if the paper has no usable text to embed.

    The embedded text combines title + abstract (better semantic signal
    than abstract alone), while the *complete* bibliographic metadata
    (authors, DOI, year, source, journal, PMID/PMCID, URLs, citation count)
    is preserved verbatim in ``EmbeddingMetadata.extra`` so a `Paper` can be
    faithfully reconstructed later for display (see
    `_reconstruct_paper_from_metadata`).

    Parameters
    ----------
    paper:
        The literature search result to convert.

    Returns
    -------
    EmbeddingMetadata | None
        The metadata ready for embedding, or ``None`` if the paper has
        neither a title nor an abstract (nothing meaningful to embed).
    """
    combined_text = " ".join(part for part in (paper.title, paper.abstract) if part).strip()
    if not combined_text:
        return None

    return EmbeddingMetadata(
        source_id=paper.uid,
        source_type=_EMBEDDING_SOURCE_TYPE,
        title=paper.title or "(untitled)",
        text=combined_text,
        chunk_index=0,
        extra={
            "abstract": paper.abstract,
            "authors": paper.authors,
            "journal": paper.journal,
            "year": paper.year,
            "doi": paper.doi,
            "pmid": paper.pmid,
            "pmcid": paper.pmcid,
            "source": paper.source,
            "citation_count": paper.citation_count,
            "is_open_access": paper.is_open_access,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "publication_type": paper.publication_type,
        },
    )


def _reconstruct_paper_from_metadata(metadata: EmbeddingMetadata) -> Paper:
    """
    Rebuild a displayable `Paper` from stored `EmbeddingMetadata`.

    Used by the Semantic Search tab, where matches come back from the
    embeddings store (possibly from a previous session) rather than from
    `st.session_state.bx_last_results`, so the original `Paper` objects are
    not necessarily still in memory.

    Parameters
    ----------
    metadata:
        The metadata previously produced by `_paper_to_embedding_metadata`.

    Returns
    -------
    Paper
        A `Paper` populated from the preserved metadata, suitable for
        reuse with the existing `_render_paper_card` UI component.
    """
    extra = metadata.extra or {}
    return Paper(
        title=metadata.title,
        abstract=extra.get("abstract", ""),
        authors=list(extra.get("authors") or []),
        journal=extra.get("journal", ""),
        year=extra.get("year"),
        doi=extra.get("doi", ""),
        pmid=extra.get("pmid", ""),
        pmcid=extra.get("pmcid", ""),
        source=extra.get("source", ""),
        citation_count=extra.get("citation_count"),
        is_open_access=bool(extra.get("is_open_access", False)),
        url=extra.get("url", ""),
        pdf_url=extra.get("pdf_url", ""),
        publication_type=extra.get("publication_type", ""),
    )


def ingest_papers_into_embeddings(papers: list[Paper]) -> dict[str, int]:
    """
    Generate and store embeddings for literature search results, skipping
    papers that already have embeddings and papers with no usable text.

    Never raises: failures for individual papers are logged and counted,
    so one bad record cannot interrupt ingestion of the rest of the batch,
    and this function can safely be called after every search without
    risking the primary search flow.

    Parameters
    ----------
    papers:
        The papers returned by `run_federated_search` (after dedupe/filter/
        rank) to embed and store.

    Returns
    -------
    dict[str, int]
        Summary counters: ``embedded``, ``skipped_duplicate``,
        ``skipped_no_text``, and ``failed``.
    """
    stats = {"embedded": 0, "skipped_duplicate": 0, "skipped_no_text": 0, "failed": 0}
    if not papers:
        return stats

    try:
        manager = _get_embedding_manager()
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not initialize the embedding manager: %s", exc)
        stats["failed"] = len(papers)
        return stats

    for paper in papers:
        metadata = _paper_to_embedding_metadata(paper)
        if metadata is None:
            stats["skipped_no_text"] += 1
            continue

        try:
            existing = manager.get_by_source(paper.uid)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Duplicate-check failed for paper uid='%s' ('%s'): %s",
                paper.uid, paper.title, exc,
            )
            existing = []

        if existing:
            stats["skipped_duplicate"] += 1
            continue

        try:
            manager.add_document(metadata)
            stats["embedded"] += 1
        except EmbeddingManagerError as exc:
            logger.warning(
                "Failed to embed paper uid='%s' ('%s'): %s",
                paper.uid, paper.title, exc,
            )
            stats["failed"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error embedding paper uid='%s' ('%s').", paper.uid, paper.title
            )
            stats["failed"] += 1

    logger.info(
        "Literature embedding ingestion complete: %d embedded, %d duplicates skipped, "
        "%d no-text skipped, %d failed (of %d total).",
        stats["embedded"], stats["skipped_duplicate"], stats["skipped_no_text"],
        stats["failed"], len(papers),
    )
    return stats


# ==============================================================================
# 7. LIGHTWEIGHT AI ANALYSIS (heuristic, no paid APIs required)
# ==============================================================================

_DISEASE_HINTS = ["cancer", "carcinoma", "diabetes", "alzheimer", "parkinson", "syndrome",
                   "disease", "disorder", "infection", "arthritis", "asthma", "stroke",
                   "obesity", "hypertension", "depression", "sclerosis"]
_GENE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\d?\b")
_DRUG_SUFFIXES = ("mab", "nib", "prazole", "statin", "cillin", "azole", "olol", "pril", "sartan")
_ORGANISM_HINTS = ["mouse", "mice", "rat", "human", "e. coli", "yeast", "drosophila", "zebrafish"]


def extract_entities(text: str) -> dict[str, list[str]]:
    """Regex/keyword-based biomedical entity extraction. Not a substitute for
    a trained NER model, but dependency-free and reasonably useful for
    surfacing likely diseases, genes, drugs, and organisms in an abstract."""
    if not text:
        return {"diseases": [], "genes": [], "proteins": [], "chemicals": [], "drugs": [], "organisms": []}

    lower = text.lower()
    diseases = sorted({h for h in _DISEASE_HINTS if h in lower})

    tokens = set(re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", text))
    genes = sorted({t for t in tokens if _GENE_PATTERN.fullmatch(t) and not t.isupper() is False and len(t) <= 8})
    drugs = sorted({t for t in tokens if t.lower().endswith(_DRUG_SUFFIXES)})
    organisms = sorted({h for h in _ORGANISM_HINTS if h in lower})

    return {
        "diseases": diseases[:10],
        "genes": genes[:10],
        "proteins": [],
        "chemicals": [],
        "drugs": drugs[:10],
        "organisms": organisms[:5],
    }


def generate_summaries(paper: Paper) -> Paper:
    """Populate short/detailed heuristic summaries and entities for a paper.
    Uses simple extractive summarization (first + highest-signal sentences)
    since no paid LLM API is assumed to be configured. If an Anthropic API
    key is available in Streamlit secrets, callers may optionally upgrade
    this with a real model call (see `generate_ai_summary_with_claude`)."""
    abstract = paper.abstract or ""
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    sentences = [s for s in sentences if len(s.split()) > 4]

    paper.short_summary = sentences[0] if sentences else "No abstract available for this paper."
    paper.detailed_summary = " ".join(sentences[:4]) if sentences else paper.short_summary
    paper.entities = extract_entities(abstract or paper.title)
    if not paper.keywords:
        paper.keywords = _extract_keywords(abstract or paper.title)
    return paper


def _extract_keywords(text: str, top_n: int = 6) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "were", "was", "are", "from",
            "have", "has", "not", "into", "using", "study", "results", "which", "these"}
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


def generate_ai_summary_with_claude(paper: Paper, api_key: str) -> Optional[str]:
    """Optional upgrade path: if the user supplies an Anthropic API key,
    generate a genuinely AI-written structured summary. Silently returns
    None (falling back to heuristic summaries) on any failure, since this
    module must not hard-depend on a paid API."""
    if not api_key or not paper.abstract:
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 400,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Summarize this biomedical abstract for a researcher. Cover: "
                        "research contribution, novelty, clinical significance, limitations, "
                        "and future work, each in one short line.\n\n"
                        f"Title: {paper.title}\nAbstract: {paper.abstract}"
                    ),
                }],
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json().get("content", [])
        text_blocks = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(text_blocks) if text_blocks else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude summary generation failed: %s", exc)
        return None


def find_similar_papers(target: Paper, pool: list[Paper], top_n: int = 5) -> list[Paper]:
    """Simple keyword-overlap similarity across title + abstract, excluding
    the target paper itself."""
    target_terms = set(re.findall(r"[a-zA-Z]{4,}", (target.title + " " + target.abstract).lower()))
    scored = []
    for p in pool:
        if p.uid == target.uid:
            continue
        terms = set(re.findall(r"[a-zA-Z]{4,}", (p.title + " " + p.abstract).lower()))
        overlap = len(target_terms & terms)
        if overlap:
            scored.append((overlap, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_n]]


# ==============================================================================
# 8. CITATION ANALYSIS
# ==============================================================================

def citation_summary(papers: list[Paper]) -> dict[str, Any]:
    counts = [p.citation_count for p in papers if p.citation_count is not None]
    influential = [p.influential_citations for p in papers if p.influential_citations is not None]
    refs = [p.reference_count for p in papers if p.reference_count is not None]
    top_cited = sorted([p for p in papers if p.citation_count], key=lambda p: -p.citation_count)[:10]

    return {
        "total_citations": sum(counts) if counts else 0,
        "avg_citations": round(sum(counts) / len(counts), 1) if counts else 0,
        "total_influential": sum(influential) if influential else 0,
        "avg_references": round(sum(refs) / len(refs), 1) if refs else 0,
        "top_cited": top_cited,
    }


# ==============================================================================
# 9. VISUALIZATIONS (Plotly)
# ==============================================================================

_PLOT_TEMPLATE = "plotly_dark"
_ACCENT = "#33d6a6"

# A literal em-dash character (not a \uXXXX escape). Defined once here so it
# can be safely referenced inside f-string expression braces (e.g.
# f"{paper.doi or _EM_DASH}") — Python versions before 3.12 disallow a
# backslash escape directly inside an f-string's {...} expression part.
_EM_DASH = "—"


def fig_publications_per_year(papers: list[Paper]) -> go.Figure:
    years = [p.year for p in papers if p.year]
    if not years:
        return _empty_fig("No year data available")
    df = pd.Series(years).value_counts().sort_index()
    fig = px.bar(x=df.index, y=df.values, labels={"x": "Year", "y": "Papers"},
                 title="Publication Trend by Year", template=_PLOT_TEMPLATE,
                 color_discrete_sequence=[_ACCENT])
    return fig


def fig_top_journals(papers: list[Paper], top_n: int = 10) -> go.Figure:
    journals = [p.journal for p in papers if p.journal]
    if not journals:
        return _empty_fig("No journal data available")
    df = pd.Series(journals).value_counts().head(top_n)
    fig = px.bar(x=df.values, y=df.index, orientation="h", labels={"x": "Papers", "y": "Journal"},
                 title="Top Journals", template=_PLOT_TEMPLATE, color_discrete_sequence=[_ACCENT])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def fig_top_authors(papers: list[Paper], top_n: int = 10) -> go.Figure:
    authors = [a for p in papers for a in p.authors]
    if not authors:
        return _empty_fig("No author data available")
    df = pd.Series(authors).value_counts().head(top_n)
    fig = px.bar(x=df.values, y=df.index, orientation="h", labels={"x": "Papers", "y": "Author"},
                 title="Top Authors", template=_PLOT_TEMPLATE, color_discrete_sequence=[_ACCENT])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def fig_keyword_frequency(papers: list[Paper], top_n: int = 15) -> go.Figure:
    kws = [k.lower() for p in papers for k in p.keywords if k]
    if not kws:
        return _empty_fig("No keyword data available")
    df = pd.Series(kws).value_counts().head(top_n)
    fig = px.treemap(names=df.index, parents=[""] * len(df), values=df.values,
                      title="Keyword Frequency", template=_PLOT_TEMPLATE,
                      color=df.values, color_continuous_scale="Tealgrn")
    return fig


def fig_open_access_pie(papers: list[Paper]) -> go.Figure:
    oa = sum(1 for p in papers if p.is_open_access)
    closed = len(papers) - oa
    fig = px.pie(names=["Open Access", "Closed / Subscription"], values=[oa, closed],
                 title="Open Access Share", template=_PLOT_TEMPLATE,
                 color_discrete_sequence=[_ACCENT, "#555"])
    return fig


def fig_citation_histogram(papers: list[Paper]) -> go.Figure:
    counts = [p.citation_count for p in papers if p.citation_count is not None]
    if not counts:
        return _empty_fig("No citation data available")
    fig = px.histogram(x=counts, nbins=20, labels={"x": "Citation Count"},
                        title="Citation Count Distribution", template=_PLOT_TEMPLATE,
                        color_discrete_sequence=[_ACCENT])
    return fig


def fig_source_breakdown(papers: list[Paper]) -> go.Figure:
    sources = [s.strip() for p in papers for s in p.source.split(",") if s.strip()]
    if not sources:
        return _empty_fig("No source data available")
    df = pd.Series(sources).value_counts()
    fig = px.bar(x=df.index, y=df.values, labels={"x": "Source", "y": "Papers"},
                 title="Results by Source", template=_PLOT_TEMPLATE,
                 color_discrete_sequence=[_ACCENT])
    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#888"))
    fig.update_layout(template=_PLOT_TEMPLATE, xaxis_visible=False, yaxis_visible=False)
    return fig


# ==============================================================================
# 10. EXPORTS
# ==============================================================================

def export_csv(papers: list[Paper]) -> bytes:
    df = pd.DataFrame([p.to_flat_dict() for p in papers])
    return df.to_csv(index=False).encode("utf-8")


def export_json(papers: list[Paper]) -> bytes:
    return json.dumps([asdict(p) for p in papers], indent=2, default=str).encode("utf-8")


def export_txt(papers: list[Paper]) -> bytes:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[{i}] {p.title}")
        lines.append(f"    Authors: {p.author_str(6)}")
        lines.append(f"    Journal: {p.journal} ({p.year or 'n.d.'})")
        lines.append(f"    DOI: {p.doi or 'N/A'}   PMID: {p.pmid or 'N/A'}")
        lines.append(f"    Citations: {p.citation_count if p.citation_count is not None else 'N/A'}")
        lines.append(f"    Abstract: {p.abstract or 'N/A'}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def export_bibtex(papers: list[Paper]) -> bytes:
    entries = []
    for p in papers:
        key = re.sub(r"[^a-zA-Z0-9]", "", (p.authors[0].split()[-1] if p.authors else "anon") + str(p.year or ""))
        entries.append(
            f"@article{{{key},\n"
            f"  title = {{{p.title}}},\n"
            f"  author = {{{' and '.join(p.authors) or 'Unknown'}}},\n"
            f"  journal = {{{p.journal}}},\n"
            f"  year = {{{p.year or ''}}},\n"
            f"  volume = {{{p.volume}}},\n"
            f"  pages = {{{p.pages}}},\n"
            f"  doi = {{{p.doi}}}\n"
            f"}}\n"
        )
    return "\n".join(entries).encode("utf-8")


def export_ris(papers: list[Paper]) -> bytes:
    entries = []
    for p in papers:
        lines = ["TY  - JOUR"]
        for a in p.authors:
            lines.append(f"AU  - {a}")
        lines.append(f"TI  - {p.title}")
        lines.append(f"JO  - {p.journal}")
        if p.year:
            lines.append(f"PY  - {p.year}")
        if p.volume:
            lines.append(f"VL  - {p.volume}")
        if p.pages:
            lines.append(f"SP  - {p.pages}")
        if p.doi:
            lines.append(f"DO  - {p.doi}")
        if p.abstract:
            lines.append(f"AB  - {p.abstract}")
        lines.append("ER  - ")
        entries.append("\n".join(lines))
    return "\n\n".join(entries).encode("utf-8")


def export_pdf_report(papers: list[Paper], title: str = "BioNexus Literature Report") -> Optional[bytes]:
    if not _HAS_FPDF:
        return None
    def cell(text: str, size: int = 10, bold: bool = False, height: int = 6) -> None:
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, height, _pdf_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    cell(title, size=16, bold=True, height=10)
    cell(f"Generated: {dt.date.today().isoformat()}  |  {len(papers)} papers", size=10)
    pdf.ln(4)

    for i, p in enumerate(papers, 1):
        cell(f"{i}. {p.title}", size=11, bold=True)
        cell(f"{p.author_str(6)} | {p.journal} ({p.year or 'n.d.'})", size=9, height=5)
        if p.doi:
            cell(f"DOI: {p.doi}", size=9, height=5)
        if p.abstract:
            cell(p.abstract[:600], size=9, height=5)
        pdf.ln(3)

    return bytes(pdf.output(dest="S"))


def _pdf_safe(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def generate_literature_review(papers: list[Paper], topic: str) -> str:
    """Auto-generate a structured literature review draft from the ranked
    result set: Introduction, Current Research, Key Findings, Research Gaps,
    Future Scope, and a formatted reference list."""
    year_span = [p.year for p in papers if p.year]
    span_str = f"{min(year_span)}–{max(year_span)}" if year_span else "recent years"
    top = papers[:15]

    intro = (
        f"This review synthesizes {len(papers)} studies retrieved on the topic of "
        f"\"{topic}\", spanning {span_str} across {len(set(p.journal for p in papers if p.journal))} "
        f"distinct journals and preprint servers."
    )

    current_research = "\n".join(
        f"- {p.title} ({p.author_str(2)}, {p.year or 'n.d.'}) — "
        f"{(p.short_summary or p.abstract[:160] or 'No summary available.')}"
        for p in top
    )

    findings = "\n".join(
        f"- {p.title}: {p.detailed_summary or p.short_summary or 'Summary unavailable.'}"
        for p in top[:8]
    )

    gaps = (
        "- Several studies rely on limited sample sizes or single-center cohorts, "
        "limiting generalizability.\n"
        "- Long-term outcome data remain sparse across the reviewed literature.\n"
        "- Mechanistic explanations are, in a subset of studies, inferred rather than "
        "directly validated."
    )

    future = (
        "- Larger, multi-center and multi-omics studies are needed to validate "
        "preliminary findings.\n"
        "- Standardized reporting of methodology and effect sizes would improve "
        "cross-study comparability.\n"
        "- Translational and clinical validation studies represent a natural next step "
        "for the more mechanistic findings identified here."
    )

    references = "\n".join(
        f"{i}. {p.author_str(10)}. {p.title}. {p.journal}"
        f"{', ' + str(p.year) if p.year else ''}."
        f"{' DOI: ' + p.doi if p.doi else ''}"
        for i, p in enumerate(papers, 1)
    )

    return (
        f"# Literature Review: {topic}\n\n"
        f"## Introduction\n{intro}\n\n"
        f"## Current Research\n{current_research}\n\n"
        f"## Important Findings\n{findings}\n\n"
        f"## Research Gaps\n{gaps}\n\n"
        f"## Future Scope\n{future}\n\n"
        f"## References\n{references}\n"
    )


# ==============================================================================
# 11. SESSION STATE / HISTORY / FAVORITES
# ==============================================================================

def _init_state() -> None:
    defaults = {
        "bx_search_history": [],       # list of {query, filters, timestamp}
        "bx_favorites": {},            # uid -> paper dict
        "bx_last_results": [],         # list of Paper
        "bx_last_status": {},
        "bx_last_query": "",
        "bx_selected_paper": None,
        "bx_claude_api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_to_history(query: str, filters: SearchFilters) -> None:
    entry = {
        "query": query,
        "filters": asdict(filters),
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.bx_search_history.insert(0, entry)
    st.session_state.bx_search_history = st.session_state.bx_search_history[:50]


def toggle_favorite(paper: Paper) -> None:
    favs = st.session_state.bx_favorites
    if paper.uid in favs:
        del favs[paper.uid]
    else:
        favs[paper.uid] = asdict(paper)


# ==============================================================================
# 12. STYLING
# ==============================================================================

def _inject_css() -> None:
    st.markdown(f"""
    <style>
      .bx-header {{
          padding: 1.25rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;
          background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
          border: 1px solid rgba(51,214,166,0.25);
      }}
      .bx-header h1 {{ margin: 0; font-size: 1.6rem; color: #e8fdf6; }}
      .bx-header p {{ margin: 0.25rem 0 0 0; color: #9fd8c8; font-size: 0.9rem; }}

      .bx-card {{
          background: #14181c; border: 1px solid #262b31; border-radius: 12px;
          padding: 1rem 1.1rem; margin-bottom: 0.85rem;
          transition: border-color 0.15s ease;
      }}
      .bx-card:hover {{ border-color: {_ACCENT}; }}
      .bx-title {{ font-size: 1.05rem; font-weight: 600; color: #f2f6f5; margin-bottom: 0.15rem; }}
      .bx-meta {{ font-size: 0.82rem; color: #9aa5a1; margin-bottom: 0.4rem; }}
      .bx-badge {{
          display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
          font-size: 0.72rem; font-weight: 600; margin-right: 0.35rem; margin-bottom: 0.25rem;
      }}
      .bx-badge-source {{ background: rgba(51,214,166,0.15); color: {_ACCENT}; }}
      .bx-badge-oa {{ background: rgba(88,166,255,0.15); color: #58a6ff; }}
      .bx-badge-cite {{ background: rgba(255,196,0,0.15); color: #ffc400; }}
      .bx-abstract {{ font-size: 0.88rem; color: #c7cdca; line-height: 1.5; }}
      .bx-metric-row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.5rem 0 1rem 0; }}
    </style>
    """, unsafe_allow_html=True)


def _render_header() -> None:
    st.markdown(f"""
    <div class="bx-header">
      <h1>🧬 {Config.APP_NAME}</h1>
      <p>AI-powered aggregation across PubMed, Europe&nbsp;PMC, CrossRef, Semantic&nbsp;Scholar,
      OpenAlex, DOAJ, bioRxiv &amp; medRxiv — merged, ranked, and analyzed.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_paper_card(paper: Paper, all_results: list[Paper], context: str = "default") -> None:
    """
    Render a single paper as an interactive card.

    Parameters
    ----------
    paper:
        The paper to render.
    all_results:
        The full result set the card belongs to, used to compute
        "similar / related papers" within the card's AI analysis popover.
    context:
        A short, unique-per-call-site label (e.g. ``"search"``,
        ``"semantic"``, ``"favorites"``) used to namespace this card's
        widget keys. The same paper can legitimately be rendered from more
        than one tab within a single script run (for example, a paper
        found via keyword search and also surfaced later via Semantic
        Search), and Streamlit requires widget keys to be unique across
        the *entire* run, not just within one tab — so `paper.uid` alone
        is not sufficient once a paper can appear in multiple places.
    """
    is_fav = paper.uid in st.session_state.bx_favorites
    widget_key_prefix = f"{context}_{paper.uid}"
    with st.container():
        st.markdown('<div class="bx-card">', unsafe_allow_html=True)

        badges = f'<span class="bx-badge bx-badge-source">{html.escape(paper.source or "Unknown")}</span>'
        if paper.is_open_access:
            badges += '<span class="bx-badge bx-badge-oa">Open Access</span>'
        if paper.citation_count is not None:
            badges += f'<span class="bx-badge bx-badge-cite">{paper.citation_count} citations</span>'

        st.markdown(f'<div class="bx-title">{html.escape(paper.title)}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="bx-meta">{html.escape(paper.author_str())} &middot; '
            f'{html.escape(paper.journal or "Unknown journal")} &middot; {paper.year or "n.d."}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(badges, unsafe_allow_html=True)

        with st.expander("Abstract & details"):
            st.markdown(f'<div class="bx-abstract">{html.escape(paper.abstract or "No abstract available.")}</div>',
                         unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"**DOI:** {paper.doi or _EM_DASH}")
                st.caption(f"**PMID:** {paper.pmid or _EM_DASH}  |  **PMCID:** {paper.pmcid or _EM_DASH}")
                st.caption(f"**Volume/Issue/Pages:** {paper.volume or _EM_DASH}/{paper.issue or _EM_DASH}/{paper.pages or _EM_DASH}")
            with col_b:
                st.caption(f"**Publisher:** {paper.publisher or _EM_DASH}")
                st.caption(f"**Type:** {paper.publication_type or _EM_DASH}")
                st.caption(f"**PDF:** {'Available' if paper.pdf_url else 'Not found'}")

            if not paper.short_summary:
                generate_summaries(paper)

            st.markdown("**AI short summary:** " + paper.short_summary)
            with st.popover("Full AI analysis"):
                st.markdown("**Detailed summary**")
                st.write(paper.detailed_summary)
                st.markdown("**Extracted entities**")
                for k, v in paper.entities.items():
                    if v:
                        st.write(f"- **{k.capitalize()}**: {', '.join(v)}")
                similar = find_similar_papers(paper, all_results)
                if similar:
                    st.markdown("**Similar / related papers**")
                    for s in similar:
                        st.write(f"- {s.title} ({s.year or 'n.d.'})")

            btn_cols = st.columns([1, 1, 2])
            with btn_cols[0]:
                if st.button("★ Favorite" if not is_fav else "★ Unfavorite",
                              key=f"fav_{widget_key_prefix}"):
                    toggle_favorite(paper)
                    st.rerun()
            with btn_cols[1]:
                if paper.url:
                    st.link_button("Open source", paper.url, key=f"open_{widget_key_prefix}")
            with btn_cols[2]:
                if paper.pdf_url:
                    st.link_button("PDF", paper.pdf_url, key=f"pdf_{widget_key_prefix}")

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 13. MAIN ENTRY POINT
# ==============================================================================

def show() -> None:
    """Public entry point — call this from app.py to render the full
    BioNexus literature search experience. No other setup required."""
    _init_state()
    _inject_css()
    _render_header()

    tabs = st.tabs(["🔍 Search", "🧠 Semantic Search", "📊 Analytics",
                     "📝 Literature Review", "🕐 History", "★ Favorites",
                     "⚙️ Settings"])

    # -------------------------------------------------------------- Search
    with tabs[0]:
        _render_search_tab()

    # ------------------------------------------------------ Semantic Search
    with tabs[1]:
        _render_semantic_search_tab()

    # ----------------------------------------------------------- Analytics
    with tabs[2]:
        _render_analytics_tab()

    # ------------------------------------------------------- Lit. Review
    with tabs[3]:
        _render_review_tab()

    # ---------------------------------------------------------- History
    with tabs[4]:
        _render_history_tab()

    # --------------------------------------------------------- Favorites
    with tabs[5]:
        _render_favorites_tab()

    # ---------------------------------------------------------- Settings
    with tabs[6]:
        _render_settings_tab()


def _render_search_tab() -> None:
    with st.form("bx_search_form"):
        query = st.text_input("Search biomedical literature",
                               placeholder="e.g. EGFR mutation lung cancer, \"BRCA1\" AND breast cancer")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            year_from = st.number_input("Year from", min_value=1900, max_value=2100, value=2015, step=1)
        with c2:
            year_to = st.number_input("Year to", min_value=1900, max_value=2100,
                                       value=dt.date.today().year, step=1)
        with c3:
            article_type = st.selectbox("Article type",
                                         ["Any", "Review", "Clinical Trial", "Meta-analysis"])
        with c4:
            sort_by = st.selectbox("Sort by",
                                    ["Most Relevant", "Newest", "Oldest", "Highest Citation Count"])

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            journal = st.text_input("Journal filter", "")
        with c6:
            author = st.text_input("Author filter", "")
        with c7:
            max_results = st.slider("Max papers", 10, 150, Config.DEFAULT_MAX_RESULTS, step=10)
        with c8:
            oa_only = st.checkbox("Open Access only")

        c9, c10, c11 = st.columns(3)
        with c9:
            free_full_text = st.checkbox("Free full text only")
        with c10:
            human_only = st.checkbox("Human studies only")
        with c11:
            animal_only = st.checkbox("Animal studies only")

        st.markdown("**Sources**")
        selected_sources = st.multiselect("Data sources", Config.SOURCES,
                                           default=Config.SOURCES)

        submitted = st.form_submit_button("🔎 Search BioNexus", use_container_width=True)

    if submitted:
        validation = validate_query(query)
        if not validation.is_valid:
            st.error(validation.message)
            return
        if not validation.looks_biomedical:
            st.warning("This query doesn't look strongly biomedical — searching anyway, "
                       "but consider adding a disease, gene, drug, or clinical term for sharper results.")

        filters = SearchFilters(
            year_from=int(year_from) if year_from else None,
            year_to=int(year_to) if year_to else None,
            journal=journal, author=author, article_type=article_type,
            open_access_only=oa_only, human_studies_only=human_only,
            animal_studies_only=animal_only, free_full_text_only=free_full_text,
            max_results=max_results, sort_by=sort_by,
        )

        parsed = parse_boolean_query(query)
        plain_query = build_plain_query(parsed)

        progress = st.progress(0, text="Contacting biomedical databases…")
        with st.spinner("Searching PubMed, Europe PMC, CrossRef, Semantic Scholar, OpenAlex, DOAJ, "
                         "bioRxiv and medRxiv…"):
            results, status = run_federated_search(plain_query, filters, selected_sources)
        progress.progress(100, text="Done")
        progress.empty()

        st.session_state.bx_last_results = results
        st.session_state.bx_last_status = status
        st.session_state.bx_last_query = query
        add_to_history(query, filters)

        if results:
            with st.spinner("Generating embeddings for semantic search…"):
                embed_stats = ingest_papers_into_embeddings(results)
            st.caption(
                f"🧠 Semantic index updated — {embed_stats['embedded']} new embedding(s) added, "
                f"{embed_stats['skipped_duplicate']} already indexed, "
                f"{embed_stats['skipped_no_text']} skipped (no title/abstract text)"
                + (f", {embed_stats['failed']} failed" if embed_stats["failed"] else "") + "."
            )

        if not results:
            st.warning("No results found across the selected sources. Try broadening filters, "
                       "removing the journal/author filter, or checking your spelling.")
        else:
            st.success(f"Found {len(results)} merged, deduplicated papers.")

        with st.expander("Source status"):
            for src, msg in status.items():
                st.write(f"**{src}**: {msg}")

    _render_results_section()


def _render_results_section() -> None:
    results: list[Paper] = st.session_state.bx_last_results
    if not results:
        st.info("Run a search above to see results here.")
        return

    st.subheader(f"Results ({len(results)})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total papers", len(results))
    m2.metric("Open access", sum(1 for p in results if p.is_open_access))
    total_cites = sum(p.citation_count or 0 for p in results)
    m3.metric("Total citations", total_cites)
    years = [p.year for p in results if p.year]
    m4.metric("Year range", f"{min(years)}–{max(years)}" if years else "—")

    export_cols = st.columns(6)
    with export_cols[0]:
        st.download_button("CSV", export_csv(results), "bionexus_results.csv", "text/csv")
    with export_cols[1]:
        st.download_button("JSON", export_json(results), "bionexus_results.json", "application/json")
    with export_cols[2]:
        st.download_button("TXT", export_txt(results), "bionexus_results.txt", "text/plain")
    with export_cols[3]:
        st.download_button("BibTeX", export_bibtex(results), "bionexus_results.bib", "text/plain")
    with export_cols[4]:
        st.download_button("RIS", export_ris(results), "bionexus_results.ris", "text/plain")
    with export_cols[5]:
        pdf_bytes = export_pdf_report(results, f"Literature Report: {st.session_state.bx_last_query}")
        if pdf_bytes:
            st.download_button("PDF", pdf_bytes, "bionexus_report.pdf", "application/pdf")
        else:
            st.caption("Install `fpdf2` for PDF export.")

    page_size = 10
    total_pages = max(1, (len(results) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start, end = (page - 1) * page_size, page * page_size

    for paper in results[start:end]:
        _render_paper_card(paper, results, context="search")


def _render_semantic_search_tab() -> None:
    """
    Render the Semantic Search section: a natural-language biomedical query
    box that retrieves the Top-K most similar previously-indexed papers via
    cosine similarity over embeddings, using the existing `embeddings`
    package (`EmbeddingManager` -> `InMemoryEmbeddingSearchEngine` ->
    cosine `SimilarityMetric`, all reused unmodified).

    Results are shown using the same `_render_paper_card` component as the
    keyword search tab, so the two experiences look and feel consistent.
    """
    st.write(
        "Search using natural language instead of keywords — e.g. "
        "*\"drugs that inhibit tumor angiogenesis\"* or "
        "*\"how does insulin resistance develop in type 2 diabetes\"*. "
        "This searches papers already indexed from prior literature searches."
    )

    try:
        manager = _get_embedding_manager()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Semantic search is currently unavailable: {exc}")
        logger.error("Failed to initialize embedding manager for semantic search: %s", exc)
        return

    indexed_count = manager.count()
    st.caption(f"📚 {indexed_count} paper embedding(s) currently indexed.")

    with st.form("bx_semantic_search_form"):
        semantic_query = st.text_input(
            "Natural-language biomedical query",
            placeholder="e.g. EGFR kinase inhibitors for non-small cell lung cancer",
        )
        top_k = st.slider("Number of results (Top-K)", min_value=1, max_value=50, value=10)
        semantic_submitted = st.form_submit_button("🧠 Semantic Search", use_container_width=True)

    if not semantic_submitted:
        return

    if not semantic_query or not semantic_query.strip():
        st.error("Please enter a natural-language query to search for.")
        return

    if indexed_count == 0:
        st.warning(
            "No papers are indexed yet. Run a keyword search in the Search tab first — "
            "results are automatically embedded and added to the semantic index."
        )
        return

    try:
        with st.spinner("Finding semantically similar papers…"):
            search_results = manager.search(
                semantic_query, top_k=top_k, source_type=_EMBEDDING_SOURCE_TYPE
            )
    except ValueError as exc:
        st.error(f"Invalid semantic search request: {exc}")
        return
    except EmbeddingManagerError as exc:
        st.error(f"Semantic search failed: {exc}")
        logger.error("Semantic search failed for query %r: %s", semantic_query, exc)
        return

    if not search_results:
        st.info("No semantically similar papers found. Try indexing more papers via a keyword search.")
        return

    st.success(f"Found {len(search_results)} semantically similar paper(s).")

    reconstructed_papers = [
        _reconstruct_paper_from_metadata(result.record.metadata) for result in search_results
    ]

    for result, paper in zip(search_results, reconstructed_papers):
        st.markdown(f"**Similarity score:** `{result.score:.4f}`  (rank #{result.rank})")
        _render_paper_card(paper, reconstructed_papers, context="semantic")


def _render_analytics_tab() -> None:
    results: list[Paper] = st.session_state.bx_last_results
    if not results:
        st.info("Run a search first to unlock analytics.")
        return

    cs = citation_summary(results)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total citations", cs["total_citations"])
    m2.metric("Avg citations/paper", cs["avg_citations"])
    m3.metric("Influential citations", cs["total_influential"])
    m4.metric("Avg references/paper", cs["avg_references"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_publications_per_year(results), use_container_width=True)
    with c2:
        st.plotly_chart(fig_open_access_pie(results), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_top_journals(results), use_container_width=True)
    with c4:
        st.plotly_chart(fig_top_authors(results), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(fig_citation_histogram(results), use_container_width=True)
    with c6:
        st.plotly_chart(fig_source_breakdown(results), use_container_width=True)

    st.plotly_chart(fig_keyword_frequency(results), use_container_width=True)

    if cs["top_cited"]:
        st.subheader("Top cited papers")
        for p in cs["top_cited"]:
            st.write(f"**{p.citation_count}** citations — {p.title} ({p.year or 'n.d.'})")


def _render_review_tab() -> None:
    results: list[Paper] = st.session_state.bx_last_results
    if not results:
        st.info("Run a search first to generate a literature review.")
        return

    st.write("Generate an academic-style literature review draft from your current search results.")
    if st.button("📝 Generate literature review"):
        for p in results[:15]:
            if not p.short_summary:
                generate_summaries(p)
        review_md = generate_literature_review(results, st.session_state.bx_last_query or "the searched topic")
        st.session_state["bx_review_md"] = review_md

    review_md = st.session_state.get("bx_review_md")
    if review_md:
        st.markdown(review_md)
        st.download_button("Download review (Markdown)", review_md.encode("utf-8"),
                            "bionexus_literature_review.md", "text/markdown")


def _render_history_tab() -> None:
    history = st.session_state.bx_search_history
    if not history:
        st.info("Your search history will appear here.")
        return

    for i, entry in enumerate(history):
        c1, c2, c3 = st.columns([4, 2, 1])
        with c1:
            st.write(f"**{entry['query']}**")
        with c2:
            st.caption(entry["timestamp"])
        with c3:
            if st.button("Repeat", key=f"hist_repeat_{i}"):
                filters = SearchFilters(**entry["filters"])
                results, status = run_federated_search(entry["query"], filters, Config.SOURCES)
                st.session_state.bx_last_results = results
                st.session_state.bx_last_status = status
                st.session_state.bx_last_query = entry["query"]
                st.rerun()

    if st.button("🗑️ Clear history"):
        st.session_state.bx_search_history = []
        st.rerun()


def _render_favorites_tab() -> None:
    favs = st.session_state.bx_favorites
    if not favs:
        st.info("Papers you favorite will be saved here for this session.")
        return

    papers = [Paper(**v) for v in favs.values()]
    for paper in papers:
        _render_paper_card(paper, papers, context="favorites")

    st.download_button("Export favorites (CSV)", export_csv(papers), "bionexus_favorites.csv", "text/csv")


def _render_settings_tab() -> None:
    st.subheader("Settings")
    st.session_state.bx_claude_api_key = st.text_input(
        "Anthropic API key (optional — upgrades AI summaries from heuristic to model-generated)",
        value=st.session_state.bx_claude_api_key, type="password",
    )
    st.caption(
        "This module works fully without an API key, using dependency-free heuristic "
        "summarization and entity extraction. Providing a key enables higher-quality, "
        "model-generated summaries via the Anthropic API."
    )
    st.divider()
    st.write(f"**Minimum query length:** {Config.MIN_QUERY_LENGTH}")
    st.write(f"**Max results per source:** {Config.DEFAULT_MAX_RESULTS}")
    st.write(f"**Request timeout:** {Config.REQUEST_TIMEOUT}s")
    st.write(f"**PDF export available:** {'Yes' if _HAS_FPDF else 'No (install fpdf2)'}")
    st.caption("All data sources are free, public, non-paid APIs.")


# ==============================================================================
# Allow standalone execution for local testing:  streamlit run literature_search.py
# ==============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title=Config.APP_NAME, page_icon="🧬", layout="wide")
    show()