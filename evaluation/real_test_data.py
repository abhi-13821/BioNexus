"""
Real Test Datasets for BioNexus Evaluation
For research paper quality metrics
"""

# 100+ Literature Search Queries (Real biomedical topics)
LITERATURE_QUERIES_100 = [
    # Cancer-related
    "EGFR mutations in non-small cell lung cancer",
    "BRCA1 and breast cancer risk",
    "PD-L1 expression in melanoma",
    "KRAS mutations in colorectal cancer",
    "TP53 tumor suppressor gene",
    "HER2 positive breast cancer treatment",
    "ALK rearrangements in lung cancer",
    "BRAF V600E mutation in melanoma",
    "NTRK gene fusions in solid tumors",
    "MET exon 14 skipping in NSCLC",
    # Cardiovascular
    "statins and cardiovascular disease prevention",
    "atrial fibrillation treatment guidelines",
    "hypertension management in elderly",
    "heart failure with preserved ejection fraction",
    "coronary artery disease risk factors",
    "PCSK9 inhibitors for cholesterol",
    "aspirin for primary prevention",
    "beta-blockers after myocardial infarction",
    "ACE inhibitors in heart failure",
    "clopidogrel resistance mechanisms",
    # Infectious diseases
    "COVID-19 vaccine efficacy variants",
    "antibiotic resistance in Staphylococcus aureus",
    "HIV treatment guidelines 2024",
    "malaria drug resistance mechanisms",
    "tuberculosis treatment regimens",
    "hepatitis C direct-acting antivirals",
    "influenza vaccine effectiveness",
    "Ebola virus pathogenesis",
    "Zika virus neurological complications",
    "Streptococcus pneumoniae vaccine",
    # Neurological
    "Alzheimer's disease biomarkers",
    "Parkinson's disease treatment",
    "multiple sclerosis immunomodulation",
    "stroke thrombolysis guidelines",
    "epilepsy drug resistance",
    "amyotrophic lateral sclerosis genetics",
    "Huntington's disease mechanisms",
    "migraine prevention strategies",
    "dementia risk factors",
    "traumatic brain injury recovery",
    # Metabolic
    "type 2 diabetes management",
    "obesity pharmacotherapy",
    "metabolic syndrome components",
    "thyroid hormone replacement therapy",
    "osteoporosis treatment guidelines",
    "vitamin D deficiency supplementation",
    "gout treatment algorithms",
    "hyperlipidemia management",
    "polycystic ovary syndrome",
    "adrenal insufficiency diagnosis",
    # Immunology
    "rheumatoid arthritis biologics",
    "inflammatory bowel disease treatment",
    "psoriasis immunopathogenesis",
    "allergic rhinitis immunotherapy",
    "systemic lupus erythematosus",
    "primary immunodeficiency disorders",
    "autoimmune hepatitis treatment",
    "sarcoidosis management",
    "urticaria pathogenesis",
    "Sjogren's syndrome diagnosis",
    # Drug discovery
    "drug repurposing for COVID-19",
    "molecular docking for kinase inhibitors",
    "QSAR models for toxicity prediction",
    "machine learning in drug discovery",
    "virtual screening techniques",
    "drug-protein interaction prediction",
    "ADMET prediction methods",
    "structure-based drug design",
    "ligand-based drug design",
    "fragment-based drug discovery",
    # Genomics
    "CRISPR-Cas9 gene editing",
    "whole genome sequencing applications",
    "RNA interference therapy",
    "gene therapy for hemophilia",
    "epigenetic modifications in cancer",
    "non-coding RNA functions",
    "DNA methylation biomarkers",
    "transcription factor regulation",
    "chromatin remodeling",
    "single-cell RNA sequencing",
    # Clinical trials
    "randomized controlled trial design",
    "clinical trial endpoints selection",
    "patient recruitment strategies",
    "adaptive trial designs",
    "ethical issues in clinical trials",
    "real-world evidence generation",
    "biomarker-driven clinical trials",
    "pragmatic clinical trials",
    "registry-based clinical trials",
    "platform clinical trials",
    # Pharmacology
    "drug-drug interaction prediction",
    "pharmacokinetics modeling",
    "pharmacodynamics optimization",
    "therapeutic drug monitoring",
    "pediatric drug dosing",
    "geriatric pharmacology",
    "drug-induced liver injury",
    "nephrotoxicity mechanisms",
    "drug safety monitoring",
    "pharmacogenomics implementation",
]

# Known relevant topics for precision/recall (gold standard)
# Each topic has a list of known relevant papers
GOLD_STANDARD = {
    "EGFR mutations in non-small cell lung cancer": [
        "EGFR mutations in lung cancer",
        "EGFR mutation testing",
        "TKIs in EGFR-mutant NSCLC",
        "Osimertinib resistance",
        "EGFR T790M mutation",
    ],
    "BRCA1 and breast cancer risk": [
        "BRCA1 mutation breast cancer",
        "BRCA1 risk assessment",
        "PARP inhibitors BRCA",
        "BRCA1 hereditary cancer",
        "BRCA1 tumor suppressor",
    ],
    # Add more gold standard topics...
}

# 50 Drugs for Drug Agent testing
DRUGS_50 = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Metformin", "Atorvastatin",
    "Omeprazole", "Losartan", "Clopidogrel", "Simvastatin", "Amoxicillin",
    "Ciprofloxacin", "Metoprolol", "Lisinopril", "Furosemide", "Levothyroxine",
    "Prednisone", "Gabapentin", "Tramadol", "Albuterol", "Pantoprazole",
    "Sertraline", "Fluoxetine", "Citalopram", "Diazepam", "Alprazolam",
    "Zolpidem", "Meloxicam", "Celecoxib", "Carvedilol", "Amlodipine",
    "Hydrochlorothiazide", "Warfarin", "Rivaroxaban", "Apixaban", "Enoxaparin",
    "Insulin", "Glipizide", "Pioglitazone", "Liraglutide", "Semaglutide",
    "Empagliflozin", "Dapagliflozin", "Canagliflozin", "Valsartan", "Telmisartan",
    "Atenolol", "Propranolol", "Duloxetine", "Venlafaxine", "Bupropion",
]

# 100 SMILES for SMILES Agent testing
SMILES_100 = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "C1=CC=CC=C1",  # Benzene
    "CCCCCC",  # Hexane
    # ... add 96 more common molecules
]