"""
Shared constants for MCP scripts.

These constants are extracted and simplified from repo/geometric-rna-design/src/constants.py
to minimize dependencies while maintaining functionality.
"""

# ==============================================================================
# RNA Nucleotide Mappings
# ==============================================================================
RNA_NUCLEOTIDES = ["A", "G", "C", "U"]
LETTER_TO_NUM = {"A": 0, "G": 1, "C": 2, "U": 3}
NUM_TO_LETTER = {0: "A", 1: "G", 2: "C", 3: "U"}

# ==============================================================================
# Secondary Structure Mappings
# ==============================================================================
DOTBRACKET_TO_NUM = {".": 0, "(": 1, ")": 2}

# Bracket pairs for pseudoknot analysis
BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">"
}

# ==============================================================================
# Default Values and Thresholds
# ==============================================================================
FILL_VALUE = 1e-5
DISTANCE_EPS = 1e-5

# Quality thresholds
RMSD_THRESHOLD = 2.0
TM_THRESHOLD = 0.45
GDT_THRESHOLD = 0.50

# GC content ranges
GC_CONTENT_MIN = 0.2
GC_CONTENT_MAX = 0.8

# Sequence length limits
MIN_SEQUENCE_LENGTH = 6
MAX_SEQUENCE_LENGTH = 1000

# ==============================================================================
# Model Configuration Defaults
# ==============================================================================
DEFAULT_MODEL_CONFIG = {
    "device": "cpu",
    "checkpoint": "gRNAde_drop3d@0.75_maxlen@500.h5",
    "ribonanza_model": "ribonanzanet.pt",
    "ribonanza_ss_model": "ribonanzanet_ss.pt"
}

# ==============================================================================
# File Paths and Extensions
# ==============================================================================
SUPPORTED_FORMATS = {
    "sequence": [".fasta", ".fa", ".seq"],
    "structure": [".pdb"],
    "data": [".csv", ".tsv", ".json"],
    "config": [".json", ".yaml", ".yml"]
}

# ==============================================================================
# RNA Chemical Properties
# ==============================================================================
PURINES = ["A", "G"]
PYRIMIDINES = ["C", "U"]

# Molecular weights (approximate, in Da)
NUCLEOTIDE_WEIGHTS = {
    "A": 331.2,
    "G": 347.2,
    "C": 307.2,
    "U": 308.2
}

# Base pairing rules
WATSON_CRICK_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G")
}

WOBBLE_PAIRS = {
    ("G", "U"), ("U", "G")
}

ALL_PAIRS = WATSON_CRICK_PAIRS | WOBBLE_PAIRS