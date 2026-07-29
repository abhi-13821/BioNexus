"""
BioNexus | SMILES Analyzer
===========================
A research-grade Streamlit application for analyzing small molecules from
SMILES strings using RDKit, with an interactive 3D molecular viewer built
on py3Dmol / 3Dmol.js.

Environment targeted by this module:
    Python 3.14, Streamlit 1.59+, RDKit 2026.3.3, py3Dmol 2.5.5

Notes on architecture
----------------------
- RDKit Mol objects are not reliably hashable/serializable, so cached
  functions accept and return plain strings (SMILES / MolBlock) rather
  than Mol objects. Mol objects are reconstructed on demand from those
  strings wherever they're needed.
- The 3D viewer is rendered as a self-contained HTML/JS fragment
  (3Dmol.js) embedded via `streamlit.components.v1.html`. Because the
  component runs inside a sandboxed iframe, mouse rotate/zoom/pan,
  atom-click labeling, and the Reset/Zoom/Fit buttons are all
  implemented in client-side JavaScript -- they do not require a
  round trip to the Streamlit/Python server. This also means RDKit
  cannot receive "on click" events back into Python; see the
  ATOM INFORMATION section below for how that limitation is handled.
- Semantic embedding support (compound name, SMILES, properties,
  formula, predicted activities, toxicity information, and
  pharmacological description) is layered on top of the existing
  analysis pipeline via the shared ``embeddings`` package. This is
  purely additive: nothing in the original analysis/rendering flow is
  modified, and all embedding features degrade gracefully if the
  ``embeddings`` package or its optional ``sentence-transformers``
  dependency is unavailable.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Crippen, Lipinski, rdMolDescriptors

try:
    import py3Dmol
except ImportError as exc:  # pragma: no cover
    py3Dmol = None

# --------------------------------------------------------------------------
# EMBEDDINGS INTEGRATION (BioNexus embeddings module)
# --------------------------------------------------------------------------
# The embeddings module (EmbeddingManager / EmbeddingStore / EmbeddingSearch /
# EmbeddingGenerator / EmbeddingCache) is already implemented, tested, and
# integrated with literature_search.py, drug_info.py, and
# knowledge_graphs.py. We reuse it here rather than building any parallel
# embedding infrastructure. Import failures are handled gracefully so the
# SMILES analyzer keeps working (minus semantic features) even if the
# optional ``sentence-transformers`` backend isn't installed in a given
# environment.
try:
    from embeddings.embedding_manager import EmbeddingManager, EmbeddingManagerError
    from embeddings.embedding_generator import (
        DEFAULT_EMBEDDING_MODEL_CONFIG,
        EmbeddingGenerationError,
        SentenceTransformerEmbeddingGenerator,
    )
    from embeddings.embedding_store import JSONEmbeddingStore
    from embeddings.embedding_search import InMemoryEmbeddingSearchEngine
    from embeddings.cache import LRUEmbeddingCache
    from embeddings.models import EmbeddingMetadata, SimilarityMetric, SimilaritySearchResult

    EMBEDDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover - degrade gracefully if package missing
    EMBEDDINGS_AVAILABLE = False

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

APP_TITLE = "BioNexus | SMILES Analyzer"

# CPK-style element coloring (hex, no leading '#', used by 3Dmol.js)
ELEMENT_COLORS: dict[str, str] = {
    "C": "808080",   # Gray
    "H": "FFFFFF",   # White
    "O": "FF0D0D",   # Red
    "N": "3050F8",   # Blue
    "S": "FFFF30",   # Yellow
    "P": "FF8000",   # Orange
    "CL": "1FF01F",  # Green
    "F": "90E050",   # Light Green
    "BR": "A62929",  # Brown
    "I": "940094",   # Purple
}
DEFAULT_ELEMENT_COLOR = "E0E0E0"  # fallback light gray for unlisted elements

# Human-readable legend labels, kept in a stable display order
LEGEND_ORDER = ["C", "H", "O", "N", "S", "P", "CL", "F", "BR", "I"]
LEGEND_NAMES = {
    "C": "Carbon", "H": "Hydrogen", "O": "Oxygen", "N": "Nitrogen",
    "S": "Sulfur", "P": "Phosphorus", "CL": "Chlorine", "F": "Fluorine",
    "BR": "Bromine", "I": "Iodine",
}

# Functional group SMARTS patterns.
# Order matters somewhat for readability only; RDKit evaluates each
# pattern independently against the whole molecule.
FUNCTIONAL_GROUPS: dict[str, str] = {
    "Carboxylic Acid": "[CX3](=O)[OX2H1]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "Amide": "[NX3][CX3](=[OX1])",
    "Aldehyde": "[CX3H1](=O)[#6,H]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Alcohol": "[CX4][OX2H]",
    "Phenol": "[OX2H][cX3]:[c]",
    "Ether": "[OD2]([#6])[#6]",
    "Primary Amine": "[NX3;H2;!$(NC=O)][#6]",
    "Secondary Amine": "[NX3;H1;!$(NC=O)]([#6])[#6]",
    "Tertiary Amine": "[NX3;H0;!$(NC=O)]([#6])([#6])[#6]",
    "Thiol": "[SX2H]",
    "Benzene Ring": "c1ccccc1",
    "Halogen": "[F,Cl,Br,I]",
}

EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
}

# Path used to persist compound embeddings across Streamlit reruns and
# sessions (JSON-file-backed store, matching the storage backend used
# elsewhere in BioNexus for research-scale corpora).
EMBEDDING_STORE_PATH = "data/smiles_analyzer_embeddings.json"

# The set of text fields generated and embedded for every analyzed
# compound, in a stable display order.
EMBEDDING_FIELDS = [
    "compound_name",
    "smiles_string",
    "molecular_properties",
    "molecular_formula",
    "predicted_activities",
    "toxicity_information",
    "pharmacological_description",
]
EMBEDDING_FIELD_LABELS = {
    "compound_name": "Compound Name",
    "smiles_string": "SMILES String",
    "molecular_properties": "Molecular Properties",
    "molecular_formula": "Molecular Formula",
    "predicted_activities": "Predicted Activities",
    "toxicity_information": "Toxicity Information",
    "pharmacological_description": "Pharmacological Description",
    "compound_profile": "Compound Profile (combined)",
}

# --------------------------------------------------------------------------
# PAGE CONFIG + THEME
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_theme() -> None:
    """Inject dark, research-grade CSS styling for the whole app."""
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0e1117;
            --bg-card: #161b22;
            --bg-card-alt: #1c2230;
            --border-color: #2b3240;
            --accent: #4f9dff;
            --accent-soft: #2a4a7a;
            --text-primary: #e6e6e6;
            --text-secondary: #9aa4b2;
            --pass: #2ecc71;
            --warn: #f4c542;
            --fail: #ff5c5c;
        }

        .stApp {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }

        h1, h2, h3, h4 {
            color: var(--text-primary) !important;
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .bx-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }

        .bx-card h4 {
            margin-top: 0;
            margin-bottom: 0.6rem;
            font-size: 1.02rem;
            color: var(--accent) !important;
        }

        .bx-metric {
            background-color: var(--bg-card-alt);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            text-align: center;
        }

        .bx-metric .label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-secondary);
        }

        .bx-metric .value {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .bx-pill {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .bx-pill.pass { background-color: rgba(46, 204, 113, 0.15); color: var(--pass); border: 1px solid var(--pass); }
        .bx-pill.warn { background-color: rgba(244, 197, 66, 0.15); color: var(--warn); border: 1px solid var(--warn); }
        .bx-pill.fail { background-color: rgba(255, 92, 92, 0.15); color: var(--fail); border: 1px solid var(--fail); }

        .bx-legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .bx-swatch {
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.25);
            display: inline-block;
        }

        .bx-fg-tag {
            display: inline-block;
            background-color: var(--accent-soft);
            color: #dbe9ff;
            border-radius: 8px;
            padding: 0.3rem 0.7rem;
            margin: 0.2rem 0.3rem 0.2rem 0;
            font-size: 0.82rem;
            border: 1px solid var(--accent);
        }

        div[data-testid="stMetric"] {
            background-color: var(--bg-card-alt);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #0b0e14;
            border-right: 1px solid var(--border-color);
        }

        .bx-sim-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.6rem;
            background-color: var(--bg-card-alt);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 0.4rem;
        }

        .bx-sim-score {
            font-weight: 600;
            color: var(--accent);
        }

        .bx-search-snippet {
            color: var(--text-secondary);
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# CORE RDKit HELPERS
# --------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validating a SMILES string."""
    is_valid: bool
    mol: Optional[Chem.Mol] = None
    canonical_smiles: str = ""
    error_message: str = ""

def validate_smiles(smiles: str) -> ValidationResult:
    """Parse and sanitize a SMILES string using RDKit.

    Returns a ValidationResult carrying either a sanitized Mol object
    and its canonical SMILES, or a human-readable error message.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        return ValidationResult(False, error_message="Please enter a SMILES string.")

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as exc:  # RDKit can raise on severely malformed input
        return ValidationResult(False, error_message=f"Parser error: {exc}")

    if mol is None:
        return ValidationResult(
            False,
            error_message=(
                "RDKit could not parse this SMILES string. Check for "
                "unbalanced rings/brackets, invalid atom symbols, or "
                "incorrect valence."
            ),
        )

    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception as exc:
        return ValidationResult(False, error_message=f"Sanitization error: {exc}")

    return ValidationResult(True, mol=mol, canonical_smiles=canonical)

@st.cache_data(show_spinner=False)
def compute_properties(canonical_smiles: str) -> dict:
    """Compute a dictionary of molecular properties for a SMILES string.

    Cached on the canonical SMILES string (a plain, hashable value) so
    repeated lookups of the same molecule are cheap.
    """
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("Invalid molecule passed to compute_properties.")

    mol_with_h = Chem.AddHs(mol)

    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Exact Molecular Weight": Descriptors.ExactMolWt(mol),
        "Number of Atoms": mol_with_h.GetNumAtoms(),
        "Number of Bonds": mol_with_h.GetNumBonds(),
        "TPSA": Descriptors.TPSA(mol),
        "LogP": Crippen.MolLogP(mol),
        "Molar Refractivity": Crippen.MolMR(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Ring Count": rdMolDescriptors.CalcNumRings(mol),
        "Formal Charge": Chem.GetFormalCharge(mol),
    }

@st.cache_data(show_spinner=False)
def compute_advanced_descriptors(canonical_smiles: str) -> dict:
    """Compute the advanced descriptor set (section 8)."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("Invalid molecule passed to compute_advanced_descriptors.")

    return {
        "TPSA": Descriptors.TPSA(mol),
        "LogP": Crippen.MolLogP(mol),
        "Fraction CSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "Heavy Atom Count": mol.GetNumHeavyAtoms(),
        "Aromatic Ring Count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Ring Count": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Hetero Atom Count": rdMolDescriptors.CalcNumHeteroatoms(mol),
    }

def classify(value: float, limit: float, warn_margin: float) -> str:
    """Classify a numeric property against a Lipinski-style limit.

    Returns "pass", "warn", or "fail" based on how far past the limit
    the value sits.
    """
    if value <= limit:
        return "pass"
    if value <= limit * (1 + warn_margin):
        return "warn"
    return "fail"

def evaluate_lipinski(props: dict) -> dict:
    """Evaluate the Lipinski Rule of Five and return a structured report."""
    mw = props["Exact Molecular Weight"]
    logp = props["LogP"]
    hbd = props["H-Bond Donors"]
    hba = props["H-Bond Acceptors"]

    rules = [
        {
            "name": "Molecular Weight <= 500 Da",
            "value": f"{mw:.2f} Da",
            "status": classify(mw, 500, 0.10),
            "explanation": (
                "Compounds heavier than ~500 Da tend to have poorer "
                "membrane permeability and oral absorption."
            ),
        },
        {
            "name": "LogP <= 5",
            "value": f"{logp:.2f}",
            "status": classify(logp, 5, 0.10),
            "explanation": (
                "High lipophilicity (LogP) is associated with poor "
                "solubility and increased risk of off-target binding."
            ),
        },
        {
            "name": "H-Bond Donors <= 5",
            "value": str(hbd),
            "status": classify(hbd, 5, 0.20),
            "explanation": (
                "Too many hydrogen bond donors reduces passive membrane "
                "permeability."
            ),
        },
        {
            "name": "H-Bond Acceptors <= 10",
            "value": str(hba),
            "status": classify(hba, 10, 0.20),
            "explanation": (
                "Too many hydrogen bond acceptors similarly limits "
                "passive membrane permeability."
            ),
        },
    ]

    violations = sum(1 for r in rules if r["status"] == "fail")
    soft_violations = sum(1 for r in rules if r["status"] == "warn")

    if violations == 0 and soft_violations == 0:
        overall = "pass"
    elif violations <= 1:
        overall = "warn"
    else:
        overall = "fail"

    return {"rules": rules, "overall": overall, "violations": violations}

def detect_functional_groups(canonical_smiles: str) -> list[str]:
    """Return the list of functional groups detected in the molecule."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return []

    detected = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            detected.append(name)
    return detected

@st.cache_data(show_spinner=False)
def generate_3d_molblock(canonical_smiles: str) -> tuple[str, str]:
    """Generate a 3D conformer for a molecule and return it as a MolBlock.

    Returns a tuple of (molblock, warning_message). warning_message is
    empty on full success, or explains which fallback strategy was used
    if the primary embedding/optimization failed.
    """
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("Invalid molecule passed to generate_3d_molblock.")

    mol = Chem.AddHs(mol)
    warning_message = ""

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_status = AllChem.EmbedMolecule(mol, params)

    if embed_status != 0:
        # Primary embedding failed -- retry with random coordinates.
        embed_status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if embed_status != 0:
            raise RuntimeError(
                "3D embedding failed even with the random-coordinate "
                "fallback. This molecule's geometry may be too "
                "constrained or degenerate to embed."
            )
        warning_message = "Primary 3D embedding failed; used random-coordinate fallback."

    # Try MMFF94 optimization first, fall back to UFF, then skip silently.
    try:
        mmff_ok = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if mmff_ok != 0:
            raise RuntimeError("MMFF did not fully converge.")
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            warning_message = (warning_message + " Used UFF force field (MMFF unavailable/failed).").strip()
        except Exception:
            warning_message = (
                warning_message + " Geometry optimization failed; showing unoptimized 3D coordinates."
            ).strip()

    molblock = Chem.MolToMolBlock(mol, kekulize=True)
    return molblock, warning_message

def build_atom_table(canonical_smiles: str) -> list[dict]:
    """Build a full per-atom information table.

    This exists because 3Dmol.js runs inside a sandboxed iframe and
    cannot send click events back to the Streamlit/Python process
    without a custom bidirectional component (out of scope here). The
    in-viewer click handler shows a label on the atom itself (see
    render_3d_viewer), and this table provides the same information
    -- element, atomic number, mass, valence, hybridization, formal
    charge, and bond count -- for every atom, in one place.
    """
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return []

    mol = Chem.AddHs(mol)
    hyb_names = {
        Chem.HybridizationType.SP: "sp",
        Chem.HybridizationType.SP2: "sp2",
        Chem.HybridizationType.SP3: "sp3",
        Chem.HybridizationType.SP3D: "sp3d",
        Chem.HybridizationType.SP3D2: "sp3d2",
        Chem.HybridizationType.S: "s",
        Chem.HybridizationType.UNSPECIFIED: "unspecified",
    }

    rows = []
    pt = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        rows.append(
            {
                "Index": atom.GetIdx(),
                "Element": symbol,
                "Element Name": pt.GetElementName(atom.GetAtomicNum()),
                "Atomic Number": atom.GetAtomicNum(),
                "Atomic Mass": round(pt.GetAtomicWeight(atom.GetAtomicNum()), 3),
                "Valence": atom.GetTotalValence(),
                "Hybridization": hyb_names.get(atom.GetHybridization(), "n/a"),
                "Formal Charge": atom.GetFormalCharge(),
                "Num Bonds": atom.GetDegree(),
                "Aromatic": atom.GetIsAromatic(),
            }
        )
    return rows

# --------------------------------------------------------------------------
# 3D VIEWER (py3Dmol / 3Dmol.js)
# --------------------------------------------------------------------------

def render_3d_viewer(molblock: str, height: int = 480) -> str:
    """Build a self-contained HTML fragment embedding a 3Dmol.js viewer.

    Ball-and-stick representation, CPK-style coloring, element symbol
    labels, dark background, and client-side controls for rotate/zoom/
    pan (native mouse controls) plus Reset / Zoom In / Zoom Out / Fit
    buttons and click-to-label atom info.
    """
    if py3Dmol is None:
        return (
            "<div style='color:#ff5c5c;padding:1rem;'>"
            "py3Dmol is not installed. Run <code>pip install py3Dmol</code> "
            "to enable the 3D viewer.</div>"
        )

    view = py3Dmol.view(width="100%", height=height)
    view.addModel(molblock, "mol")

    # Ball-and-stick, colored per element to approximate CPK coloring.
    view.setStyle({}, {"stick": {"radius": 0.14}, "sphere": {"scale": 0.28}})
    for element, color_hex in ELEMENT_COLORS.items():
        sel = {"elem": element.capitalize() if len(element) == 1 else element}
        # 3Dmol.js element selectors are case sensitive for two-letter
        # symbols (Cl, Br); normalize accordingly.
        elem_map = {"CL": "Cl", "BR": "Br"}
        sel["elem"] = elem_map.get(element, element.capitalize())
        view.setStyle(
            sel,
            {
                "stick": {"radius": 0.14, "color": f"0x{color_hex}"},
                "sphere": {"scale": 0.28, "color": f"0x{color_hex}"},
            },
        )

    view.setBackgroundColor("#1a1a1a")

    # Element-symbol labels directly on each atom, readable on dark bg.
    view.addPropertyLabels(
        "elem",
        {},
        {
            "fontColor": "white",
            "fontSize": 12,
            "showBackground": True,
            "backgroundColor": "black",
            "backgroundOpacity": 0.55,
            "alignment": "center",
        },
    )

    # Click-to-toggle a detailed label on any atom (client-side only --
    # see build_atom_table() docstring for why this can't round-trip
    # to the Python/Streamlit server).
    click_callback = """
    function(atom, viewer, event, container) {
        if (!atom.bx_label) {
            var text = atom.elem + " | idx " + atom.serial +
                       " | bonds " + (atom.bonds ? atom.bonds.length : "?");
            atom.bx_label = viewer.addLabel(text, {
                position: atom,
                backgroundColor: "#111111",
                backgroundOpacity: 0.85,
                fontColor: "#4f9dff",
                fontSize: 13,
                borderThickness: 1,
                borderColor: "#4f9dff"
            });
        } else {
            viewer.removeLabel(atom.bx_label);
            delete atom.bx_label;
        }
        viewer.render();
    }
    """
    view.setClickable({}, True, click_callback)
    view.zoomTo()

    try:
        viewer_html = view._make_html()
    except AttributeError:
        # Extremely defensive fallback in case a future py3Dmol release
        # renames the internal HTML export method.
        viewer_html = f"<div>{view.js()}</div>"

    control_bar = """
    <div style="display:flex; gap:0.5rem; margin-top:0.6rem; flex-wrap: wrap;">
        <button onclick="bx_resetView()" style="{btn}">Reset View</button>
        <button onclick="bx_zoom(1.25)" style="{btn}">Zoom In</button>
        <button onclick="bx_zoom(0.8)" style="{btn}">Zoom Out</button>
        <button onclick="bx_fit()" style="{btn}">Fit Molecule</button>
    </div>
    <script>
        function bx_getViewer() {
            var keys = Object.keys(window).filter(k => k.startsWith("viewer_"));
            if (keys.length === 0) { return null; }
            return window[keys[keys.length - 1]];
        }
        function bx_resetView() {
            var v = bx_getViewer();
            if (v) { v.zoomTo(); v.render(); }
        }
        function bx_zoom(factor) {
            var v = bx_getViewer();
            if (v) { v.zoom(factor, 200); }
        }
        function bx_fit() {
            var v = bx_getViewer();
            if (v) { v.zoomTo(); v.render(); }
        }
    </script>
    """.replace(
        "{btn}",
        (
            "background:#1c2230;color:#e6e6e6;border:1px solid #2b3240;"
            "border-radius:8px;padding:0.4rem 0.9rem;cursor:pointer;"
            "font-size:0.82rem;"
        ),
    )

    return viewer_html + control_bar

# --------------------------------------------------------------------------
# UI RENDERING HELPERS
# --------------------------------------------------------------------------

def card_open(title: str) -> None:
    st.markdown(f'<div class="bx-card"><h4>{title}</h4>', unsafe_allow_html=True)

def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

def metric_grid(props: dict, columns: int = 4) -> None:
    """Render a responsive grid of property metric cards."""
    items = list(props.items())
    cols = st.columns(columns)
    for i, (label, value) in enumerate(items):
        with cols[i % columns]:
            display_value = f"{value:.3f}" if isinstance(value, float) else str(value)
            st.markdown(
                f"""
                <div class="bx-metric">
                    <div class="label">{label}</div>
                    <div class="value">{display_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

def render_legend() -> None:
    for key in LEGEND_ORDER:
        color = ELEMENT_COLORS[key]
        name = LEGEND_NAMES[key]
        st.markdown(
            f"""
            <div class="bx-legend-item">
                <span class="bx-swatch" style="background:#{color};"></span>
                {name} ({key.capitalize() if len(key) > 1 else key})
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_lipinski(report: dict) -> None:
    pill_class = report["overall"]
    pill_text = {"pass": "PASS", "warn": "WARNING", "fail": "FAIL"}[pill_class]
    st.markdown(
        f'<span class="bx-pill {pill_class}">{pill_text}</span>'
        f'&nbsp;Overall Lipinski Rule of Five assessment '
        f'({report["violations"]} hard violation(s))',
        unsafe_allow_html=True,
    )
    st.write("")
    for rule in report["rules"]:
        cls = rule["status"]
        text = {"pass": "PASS", "warn": "WARNING", "fail": "FAIL"}[cls]
        st.markdown(
            f'<span class="bx-pill {cls}">{text}</span>'
            f'<strong>{rule["name"]}</strong> &mdash; {rule["value"]}',
            unsafe_allow_html=True,
        )
        st.caption(rule["explanation"])

def render_functional_groups(groups: list[str]) -> None:
    if not groups:
        st.info("No common functional groups were detected in this molecule.")
        return
    tags_html = "".join(f'<span class="bx-fg-tag">{g}</span>' for g in groups)
    st.markdown(tags_html, unsafe_allow_html=True)

def render_downloads(canonical_smiles: str, molblock: Optional[str]) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download SMILES (.smi)",
            data=canonical_smiles,
            file_name="molecule.smi",
            mime="chemical/x-daylight-smiles",
            use_container_width=True,
        )
    with col2:
        if molblock:
            st.download_button(
                "Download MOL file (.mol)",
                data=molblock,
                file_name="molecule.mol",
                mime="chemical/x-mdl-molfile",
                use_container_width=True,
            )
        else:
            st.button("MOL unavailable", disabled=True, use_container_width=True)
    with col3:
        if molblock:
            sdf_content = molblock + "\n$$$$\n"
            st.download_button(
                "Download SDF (.sdf)",
                data=sdf_content,
                file_name="molecule.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True,
            )
        else:
            st.button("SDF unavailable", disabled=True, use_container_width=True)

# --------------------------------------------------------------------------
# EMBEDDINGS INTEGRATION HELPERS
# --------------------------------------------------------------------------
# Everything below is additive: it generates embeddings for a compound
# after analysis, stores them via EmbeddingManager (deduplicated, with
# caching), and exposes semantic search + "Similar Compounds" (cosine
# similarity) UI. None of it is required for the core analysis flow above
# to keep working.

if EMBEDDINGS_AVAILABLE:

    @st.cache_resource(show_spinner=False)
    def get_embedding_manager() -> "EmbeddingManager":
        """Return a process-wide singleton EmbeddingManager.

        Cached via ``st.cache_resource`` so the underlying model, on-disk
        JSON store, and in-memory LRU cache are all constructed once per
        Streamlit process and reused across reruns and sessions -- this
        is what makes "use cached embeddings whenever available" actually
        hold across interactions, not just within a single script run.
        """
        generator = SentenceTransformerEmbeddingGenerator(DEFAULT_EMBEDDING_MODEL_CONFIG)
        store = JSONEmbeddingStore(storage_path=EMBEDDING_STORE_PATH)
        cache = LRUEmbeddingCache(max_size=20_000)
        search_engine = InMemoryEmbeddingSearchEngine(
            generator, store, metric=SimilarityMetric.COSINE
        )
        return EmbeddingManager(
            generator=generator,
            store=store,
            cache=cache,
            search_engine=search_engine,
        )

def build_predicted_activities_text(groups: list[str], props: dict) -> str:
    """Produce a short, heuristic predicted-activity summary for embedding.

    This is a lightweight, rule-of-thumb summary derived from detected
    functional groups and basic physicochemical properties -- it is not
    a validated bioactivity prediction model. It exists to give the
    embedding pipeline meaningful, differentiated text to encode so that
    semantic search and "Similar Compounds" can group molecules with
    related structural motifs.
    """
    if not groups:
        return (
            "No characteristic functional groups were detected, so no "
            "structural activity motifs are flagged for this compound. "
            "This is a heuristic structural note, not a validated "
            "bioactivity prediction."
        )

    notes = []
    group_set = set(groups)
    if "Carboxylic Acid" in group_set:
        notes.append("acidic/carboxylic motifs often seen in anti-inflammatory agents")
    if {"Primary Amine", "Secondary Amine", "Tertiary Amine"} & group_set:
        notes.append("basic amine motifs common in CNS-active and antihistaminergic compounds")
    if "Amide" in group_set:
        notes.append("amide linkages typical of peptide-like or analgesic scaffolds")
    if "Phenol" in group_set:
        notes.append("phenolic motifs associated with antioxidant and antipyretic activity")
    if "Benzene Ring" in group_set:
        notes.append("aromatic ring system consistent with broad receptor-binding scaffolds")
    if "Ester" in group_set:
        notes.append("ester functionality often used as a prodrug or metabolically labile group")
    if "Halogen" in group_set:
        notes.append("halogenation, frequently used to tune potency and metabolic stability")

    body = "; ".join(notes) if notes else "structural motifs with no strong heuristic match"
    return (
        f"Detected functional groups: {', '.join(groups)}. Heuristic structural "
        f"activity notes: {body}. This is a rule-based structural summary for "
        f"research triage only and is not a validated bioactivity prediction."
    )

def build_toxicity_text(props: dict, lipinski_report: dict, groups: list[str]) -> str:
    """Produce a short, heuristic toxicity/structural-alert summary for embedding.

    Flags common coarse structural alerts (reactive carbonyls, halogen
    loading, high lipophilicity) and Lipinski violations. This is a
    heuristic note for research triage, not a certified toxicological
    assessment.
    """
    flags = []
    group_set = set(groups)

    if "Aldehyde" in group_set:
        flags.append("aldehyde group (potentially reactive/electrophilic)")
    if group_set.intersection({"Halogen"}):
        flags.append("halogen substitution (monitor for metabolic/toxicological liability)")
    if props.get("LogP", 0) > 5:
        flags.append(f"elevated lipophilicity (LogP={props['LogP']:.2f}, risk of off-target binding)")
    if props.get("Exact Molecular Weight", 0) > 500:
        flags.append(f"high molecular weight ({props['Exact Molecular Weight']:.1f} Da)")
    if lipinski_report["violations"] > 0:
        flags.append(
            f"{lipinski_report['violations']} Lipinski Rule of Five violation(s)"
        )

    if not flags:
        return (
            "No coarse structural toxicity alerts were flagged for this compound "
            "based on functional groups, lipophilicity, or Lipinski Rule of Five "
            "compliance. This is a heuristic screening note, not a certified "
            "toxicological assessment, and does not replace experimental "
            "toxicology or regulatory review."
        )

    return (
        "Heuristic toxicity screening notes: " + "; ".join(flags) + ". "
        "This is a coarse, rule-based screening note for research triage only, "
        "not a certified toxicological assessment, and does not replace "
        "experimental toxicology or regulatory review."
    )

def build_pharmacological_description(
    props: dict, lipinski_report: dict, groups: list[str], compound_name: str
) -> str:
    """Produce a short pharmacological/druglikeness summary paragraph for embedding."""
    druglikeness = {
        "pass": "favorable oral druglikeness by Lipinski's Rule of Five",
        "warn": "borderline oral druglikeness, with minor Lipinski Rule of Five deviations",
        "fail": "poor oral druglikeness, with multiple Lipinski Rule of Five violations",
    }[lipinski_report["overall"]]

    groups_text = ", ".join(groups) if groups else "no strongly characteristic functional groups"

    return (
        f"{compound_name} is a small molecule with formula {props.get('Formula', 'n/a')} "
        f"and molecular weight {props.get('Exact Molecular Weight', 0):.2f} Da. "
        f"It exhibits {druglikeness} (LogP={props.get('LogP', 0):.2f}, "
        f"TPSA={props.get('TPSA', 0):.2f}, H-bond donors={props.get('H-Bond Donors', 0)}, "
        f"H-bond acceptors={props.get('H-Bond Acceptors', 0)}). "
        f"Structurally it contains {groups_text}. This description is a "
        f"data-derived research summary intended for semantic search and "
        f"compound triage, not a clinical or regulatory characterization."
    )

def build_compound_field_texts(
    compound_name: str,
    canonical_smiles: str,
    props: dict,
    lipinski_report: dict,
    groups: list[str],
) -> dict[str, str]:
    """Assemble the seven text fields to embed for a compound."""
    properties_text = "; ".join(
        f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in props.items()
    )

    return {
        "compound_name": compound_name,
        "smiles_string": canonical_smiles,
        "molecular_properties": properties_text,
        "molecular_formula": str(props.get("Formula", "")),
        "predicted_activities": build_predicted_activities_text(groups, props),
        "toxicity_information": build_toxicity_text(props, lipinski_report, groups),
        "pharmacological_description": build_pharmacological_description(
            props, lipinski_report, groups, compound_name
        ),
    }

def store_compound_embeddings(
    manager: "EmbeddingManager",
    compound_name: str,
    canonical_smiles: str,
    field_texts: dict[str, str],
) -> dict[str, str]:
    """Embed and store each compound field, skipping unchanged duplicates.

    For every field, a stable ``source_id`` of
    ``"<canonical_smiles>::<field>"`` is used. Before storing, existing
    records under that ``source_id`` are checked: if an identical-text
    record already exists, storage (and therefore vector duplication) is
    skipped and the cached vector is reused; only genuinely new or
    changed text is embedded and persisted. A combined "compound_profile"
    record is also stored, used as the anchor for "Similar Compounds".

    Returns
    -------
    dict[str, str]
        A status per field: "stored", "skipped (duplicate)", or an error
        message, keyed by field name (plus "compound_profile").
    """
    statuses: dict[str, str] = {}

    all_fields = dict(field_texts)
    all_fields["compound_profile"] = " | ".join(
        f"{EMBEDDING_FIELD_LABELS[key]}: {value}" for key, value in field_texts.items()
    )

    for field_name, text in all_fields.items():
        source_id = f"{canonical_smiles}::{field_name}"
        try:
            existing = manager.get_by_source(source_id)
        except Exception:
            existing = []

        if any(record.metadata.text == text for record in existing):
            statuses[field_name] = "skipped (duplicate)"
            continue

        metadata = EmbeddingMetadata(
            source_id=source_id,
            source_type=field_name,
            title=compound_name,
            text=text,
            chunk_index=0,
            extra={"canonical_smiles": canonical_smiles},
        )
        try:
            manager.add_document(metadata, use_cache=True)
            statuses[field_name] = "stored"
        except EmbeddingManagerError as exc:
            statuses[field_name] = f"error: {exc}"

    return statuses

def render_embedding_status(statuses: dict[str, str]) -> None:
    """Render a compact status line for the embedding storage pass."""
    stored = sum(1 for v in statuses.values() if v == "stored")
    skipped = sum(1 for v in statuses.values() if v.startswith("skipped"))
    errors = {k: v for k, v in statuses.items() if v.startswith("error")}

    st.caption(
        f"Embeddings: {stored} newly stored, {skipped} unchanged (deduplicated, "
        f"cached vector reused)."
    )
    if errors:
        for field_name, message in errors.items():
            st.warning(f"Embedding for '{EMBEDDING_FIELD_LABELS.get(field_name, field_name)}' failed: {message}")

def render_similar_compounds(manager: "EmbeddingManager", canonical_smiles: str, top_k: int = 5) -> None:
    """Render a 'Similar Compounds' section using cosine similarity.

    Uses the stored 'compound_profile' embedding for the current
    compound as the query vector against all other stored compound
    profiles, ranking by cosine similarity (the search engine's
    configured metric).
    """
    profile_id = f"{canonical_smiles}::compound_profile"
    try:
        results: list["SimilaritySearchResult"] = manager.find_similar(profile_id, top_k=top_k * 4)
    except EmbeddingManagerError as exc:
        st.info(f"Similar compounds are not available yet: {exc}")
        return

    # Restrict to other compounds' profile embeddings, deduplicated by
    # canonical SMILES, excluding the current compound itself.
    seen_smiles = {canonical_smiles}
    shown = 0
    for result in results:
        if result.record.metadata.source_type != "compound_profile":
            continue
        other_smiles = result.record.metadata.extra.get("canonical_smiles", "")
        if not other_smiles or other_smiles in seen_smiles:
            continue
        seen_smiles.add(other_smiles)

        st.markdown(
            f"""
            <div class="bx-sim-row">
                <div>
                    <strong>{result.record.metadata.title}</strong><br/>
                    <code>{other_smiles}</code>
                </div>
                <div class="bx-sim-score">{result.score:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load this compound", key=f"load_similar_{other_smiles}_{shown}"):
            st.session_state["smiles_override"] = other_smiles
            st.rerun()

        shown += 1
        if shown >= top_k:
            break

    if shown == 0:
        st.info(
            "No other analyzed compounds are similar enough yet. Analyze more "
            "molecules to populate this section."
        )

def render_semantic_search(manager: "EmbeddingManager") -> None:
    """Render the semantic (meaning-based) compound search UI."""
    st.caption(
        "Search previously analyzed compounds by meaning rather than exact "
        "text -- e.g. \"pain relief small molecule\" or \"halogenated aromatic "
        "compound\"."
    )
    query = st.text_input("Semantic search query", key="bx_semantic_query", placeholder="e.g. anti-inflammatory carboxylic acid")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        field_filter = st.selectbox(
            "Search within",
            options=["All fields"] + [EMBEDDING_FIELD_LABELS[f] for f in EMBEDDING_FIELDS],
            key="bx_semantic_field_filter",
        )
    with col_b:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5, key="bx_semantic_top_k")

    if not query.strip():
        return

    source_type = None
    if field_filter != "All fields":
        reverse_lookup = {v: k for k, v in EMBEDDING_FIELD_LABELS.items()}
        source_type = reverse_lookup.get(field_filter)

    try:
        results = manager.search(query.strip(), top_k=int(top_k), source_type=source_type)
    except EmbeddingManagerError as exc:
        st.error(f"Semantic search failed: {exc}")
        return

    if not results:
        st.info("No matching compounds found yet. Analyze and store more compounds first.")
        return

    for result in results:
        rec = result.record
        smiles_value = rec.metadata.extra.get("canonical_smiles", "")
        snippet = rec.metadata.text
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        st.markdown(
            f"""
            <div class="bx-sim-row">
                <div style="flex:1;">
                    <strong>{rec.metadata.title}</strong>
                    &nbsp;<span class="bx-pill pass">{EMBEDDING_FIELD_LABELS.get(rec.metadata.source_type, rec.metadata.source_type)}</span><br/>
                    <span class="bx-search-snippet">{snippet}</span>
                </div>
                <div class="bx-sim-score">{result.score:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if smiles_value:
            if st.button("Load this compound", key=f"load_search_{rec.embedding_id}"):
                st.session_state["smiles_override"] = smiles_value
                st.rerun()

# --------------------------------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------------------------------

def main() -> None:
    inject_theme()

    st.title("🧬 BioNexus")
    st.caption("Molecular structure analysis and 3D visualization, powered by RDKit")

    with st.sidebar:
        st.subheader("Input")
        example_choice = st.selectbox(
            "Load an example molecule",
            options=["-- Select --"] + list(EXAMPLE_MOLECULES.keys()),
        )
        default_smiles = (
            EXAMPLE_MOLECULES[example_choice] if example_choice != "-- Select --" else ""
        )
        # An embedding-driven "Load this compound" action (from semantic
        # search or Similar Compounds) can set this override; it takes
        # precedence over the example dropdown for one rerun.
        override_smiles = st.session_state.pop("smiles_override", None)
        smiles_input = st.text_input(
            "SMILES string",
            value=override_smiles or default_smiles,
            placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
        )
        compound_name_input = st.text_input(
            "Compound name (optional)",
            value=example_choice if example_choice != "-- Select --" else "",
            placeholder="e.g. Aspirin",
            help="Used to label this compound's stored embeddings and search results.",
        )
        st.divider()
        st.subheader("3D Viewer Settings")
        viewer_height = st.slider("Viewer height (px)", 320, 800, 480, step=20)
        st.divider()
        st.caption(
            "Atom-click details are shown directly on the 3D structure "
            "(client-side) and in the Atom Information table, since "
            "3Dmol.js cannot send click events back to the Python "
            "runtime without a custom bidirectional component."
        )

        if EMBEDDINGS_AVAILABLE:
            st.divider()
            st.subheader("Semantic Compound Search")
            try:
                manager = get_embedding_manager()
                render_semantic_search(manager)
            except Exception as exc:
                st.warning(f"Semantic search is temporarily unavailable: {exc}")

    if not smiles_input.strip():
        st.info("Enter a SMILES string in the sidebar to begin analysis.")
        return

    result = validate_smiles(smiles_input)

    if not result.is_valid:
        st.error(f"❌ Invalid SMILES: {result.error_message}")
        return

    canonical_smiles = result.canonical_smiles
    st.success(f"✅ Valid molecule. Canonical SMILES: `{canonical_smiles}`")

    compound_name = compound_name_input.strip() or "Unnamed Compound"

    # ---- Properties -----------------------------------------------------
    try:
        props = compute_properties(canonical_smiles)
    except Exception as exc:
        st.error(f"Failed to compute molecular properties: {exc}")
        return

    card_open("Molecular Properties")
    metric_grid(props, columns=4)
    card_close()

    # ---- 3D structure + viewer -------------------------------------------
    molblock = None
    viewer_warning = ""
    try:
        molblock, viewer_warning = generate_3d_molblock(canonical_smiles)
    except Exception as exc:
        st.warning(
            f"⚠️ 3D structure generation failed: {exc}. "
            "Properties, descriptors, and 2D-derived analyses below are "
            "still fully available."
        )

    left, right = st.columns([2, 1])

    with left:
        card_open("3D Molecular Viewer")
        if viewer_warning:
            st.caption(f"⚠️ {viewer_warning}")
        if molblock:
            html_fragment = render_3d_viewer(molblock, height=viewer_height)
            components.html(html_fragment, height=viewer_height + 60, scrolling=False)
        else:
            st.warning("3D viewer unavailable for this molecule.")
        card_close()

    with right:
        card_open("Legend")
        render_legend()
        card_close()

    # ---- Atom information table ------------------------------------------
    card_open("Atom Information")
    st.caption(
        "Click any atom in the 3D viewer above to toggle an inline label "
        "with its element, index, and bond count. Full per-atom detail "
        "for every atom in the molecule (including hydrogens) is listed "
        "below."
    )
    atom_rows = build_atom_table(canonical_smiles)
    if atom_rows:
        st.dataframe(atom_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No atom data available.")
    card_close()

    # ---- Advanced descriptors ---------------------------------------------
    try:
        advanced = compute_advanced_descriptors(canonical_smiles)
        card_open("Molecular Descriptors")
        metric_grid(advanced, columns=4)
        card_close()
    except Exception as exc:
        st.warning(f"Could not compute advanced descriptors: {exc}")

    # ---- Lipinski Rule of Five ----------------------------------------------
    card_open("Lipinski Rule of Five")
    lipinski_report = evaluate_lipinski(props)
    render_lipinski(lipinski_report)
    card_close()

    # ---- Functional groups ---------------------------------------------------
    card_open("Functional Group Detection")
    groups = detect_functional_groups(canonical_smiles)
    render_functional_groups(groups)
    card_close()

    # ---- Embeddings: generate + store, semantic search, similar compounds ----
    if EMBEDDINGS_AVAILABLE:
        try:
            manager = get_embedding_manager()
            field_texts = build_compound_field_texts(
                compound_name, canonical_smiles, props, lipinski_report, groups
            )
            statuses = store_compound_embeddings(manager, compound_name, canonical_smiles, field_texts)

            card_open("Compound Embeddings")
            render_embedding_status(statuses)
            with st.expander("View embedded text fields"):
                for field_name in EMBEDDING_FIELDS:
                    st.markdown(f"**{EMBEDDING_FIELD_LABELS[field_name]}**")
                    st.caption(field_texts[field_name])
            card_close()

            card_open("Similar Compounds")
            st.caption(
                "Ranked by cosine similarity between this compound's combined "
                "embedding profile and other previously analyzed compounds."
            )
            render_similar_compounds(manager, canonical_smiles, top_k=5)
            card_close()
        except Exception as exc:
            st.warning(f"Embedding features are temporarily unavailable: {exc}")
    else:
        st.info(
            "The BioNexus embeddings module is not available in this "
            "environment, so semantic search and Similar Compounds are "
            "disabled. Core SMILES analysis above is unaffected."
        )

    # ---- Downloads -------------------------------------------------------------
    card_open("Downloads")
    render_downloads(canonical_smiles, molblock)
    card_close()

    st.caption("BioNexus SMILES Analyzer — built with Streamlit, RDKit, and 3Dmol.js.")

def show():
    main()

if __name__ == "__main__":
    main()