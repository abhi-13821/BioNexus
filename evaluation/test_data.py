"""
Test datasets for BioNexus Evaluation
"""

# Test queries for Literature Agent
LITERATURE_TEST_QUERIES = [
    "EGFR mutations in lung cancer",
    "BRCA1 breast cancer treatment",
    "COVID-19 vaccine efficacy",
    "Alzheimer's disease biomarkers",
    "CRISPR gene editing",
    "PD-1 immunotherapy",
    "metformin type 2 diabetes",
    "STAT3 signaling pathway",
    "TP53 tumor suppressor",
    "CAR-T cell therapy"
]

# Test drugs for Drug Agent
DRUG_TEST_LIST = [
    "Aspirin",
    "Paracetamol",
    "Ibuprofen",
    "Metformin",
    "Atorvastatin",
    "Omeprazole",
    "Losartan",
    "Clopidogrel",
    "Simvastatin",
    "Amoxicillin"
]

# Test SMILES strings for SMILES Agent
SMILES_TEST_LIST = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CCCCC",  # Pentane
    "C1=CC=CC=C1",  # Benzene
    "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Another ibuprofen variant
    "CC1=CC=CC=C1C(=O)O",  # Salicylic acid
    "CC(=O)N1C=CC=C1",  # N-acetylpyrrole
    "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
    "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid
]