"""
Shared utility functions for MCP scripts.

General utility functions that are used across multiple scripts,
extracted and simplified from repo code to minimize dependencies.
"""

import random
from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from .constants import (
    LETTER_TO_NUM, NUM_TO_LETTER, GC_CONTENT_MIN, GC_CONTENT_MAX,
    MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, RNA_NUCLEOTIDES
)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_rna_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate RNA sequence contains only valid nucleotides.

    Args:
        sequence: RNA sequence string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sequence:
        return False, "Empty sequence"

    if len(sequence) < MIN_SEQUENCE_LENGTH:
        return False, f"Sequence too short (minimum {MIN_SEQUENCE_LENGTH} nucleotides)"

    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return False, f"Sequence too long (maximum {MAX_SEQUENCE_LENGTH} nucleotides)"

    invalid_chars = set(sequence.upper()) - set(RNA_NUCLEOTIDES)
    if invalid_chars:
        return False, f"Invalid nucleotides: {sorted(invalid_chars)}"

    return True, "Valid RNA sequence"


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of RNA sequence."""
    if not sequence:
        return 0.0

    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def calculate_nucleotide_composition(sequence: str) -> Dict[str, float]:
    """Calculate nucleotide composition statistics."""
    if not sequence:
        return {nuc: 0.0 for nuc in RNA_NUCLEOTIDES}

    sequence = sequence.upper()
    length = len(sequence)

    composition = {}
    for nuc in RNA_NUCLEOTIDES:
        composition[nuc] = sequence.count(nuc) / length

    return composition


def sequences_to_indices(sequences: List[str]) -> List[List[int]]:
    """Convert RNA sequences to numerical indices."""
    indices = []
    for seq in sequences:
        seq_indices = [LETTER_TO_NUM[nuc] for nuc in seq.upper() if nuc in LETTER_TO_NUM]
        indices.append(seq_indices)
    return indices


def indices_to_sequences(indices: List[List[int]]) -> List[str]:
    """Convert numerical indices back to RNA sequences."""
    sequences = []
    for seq_indices in indices:
        sequence = ''.join([NUM_TO_LETTER[idx] for idx in seq_indices if idx in NUM_TO_LETTER])
        sequences.append(sequence)
    return sequences


def filter_sequences_by_quality(sequences: List[str],
                               min_gc: float = GC_CONTENT_MIN,
                               max_gc: float = GC_CONTENT_MAX,
                               min_length: Optional[int] = None,
                               max_length: Optional[int] = None) -> List[str]:
    """
    Filter sequences by quality criteria.

    Args:
        sequences: List of RNA sequences
        min_gc: Minimum GC content
        max_gc: Maximum GC content
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        Filtered list of sequences
    """
    filtered = []

    for seq in sequences:
        # Validate sequence
        is_valid, _ = validate_rna_sequence(seq)
        if not is_valid:
            continue

        # Check GC content
        gc_content = calculate_gc_content(seq)
        if not (min_gc <= gc_content <= max_gc):
            continue

        # Check length constraints
        if min_length and len(seq) < min_length:
            continue
        if max_length and len(seq) > max_length:
            continue

        filtered.append(seq)

    return filtered


def calculate_sequence_diversity(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for a set of sequences.

    Returns:
        Dict with diversity statistics
    """
    if not sequences:
        return {"unique_count": 0, "diversity_ratio": 0.0, "avg_pairwise_distance": 0.0}

    # Count unique sequences
    unique_sequences = set(sequences)
    diversity_ratio = len(unique_sequences) / len(sequences)

    # Calculate average pairwise Hamming distance (sample if too many sequences)
    if len(sequences) > 100:
        # Sample pairs to avoid O(n^2) computation
        sample_pairs = min(1000, len(sequences) * (len(sequences) - 1) // 2)
        sampled_seqs = random.sample(sequences, min(50, len(sequences)))
        distances = []
        for i, seq1 in enumerate(sampled_seqs):
            for j, seq2 in enumerate(sampled_seqs[i+1:], i+1):
                if len(seq1) == len(seq2):
                    dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
                    distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
    else:
        # Calculate for all pairs
        distances = []
        for i, seq1 in enumerate(sequences):
            for j, seq2 in enumerate(sequences[i+1:], i+1):
                if len(seq1) == len(seq2):
                    dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
                    distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0

    return {
        "total_count": len(sequences),
        "unique_count": len(unique_sequences),
        "diversity_ratio": diversity_ratio,
        "avg_pairwise_distance": float(avg_distance)
    }


def create_timestamp_string() -> str:
    """Create standardized timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_execution_time(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """Format execution time in human-readable format."""
    if end_time is None:
        end_time = datetime.now()

    duration = end_time - start_time
    total_seconds = int(duration.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def merge_configs(default_config: Dict[str, Any],
                 user_config: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Merge configuration dictionaries with priority order:
    1. kwargs (highest priority)
    2. user_config
    3. default_config (lowest priority)
    """
    merged = default_config.copy()

    if user_config:
        merged.update(user_config)

    # Override with any kwargs
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value

    return merged


def find_model_checkpoint(model_name: str, search_paths: List[Union[str, Path]]) -> Optional[Path]:
    """
    Search for model checkpoint in multiple possible locations.

    Args:
        model_name: Name of model file
        search_paths: List of directories to search

    Returns:
        Path to model file if found, None otherwise
    """
    for search_path in search_paths:
        path = Path(search_path) / model_name
        if path.exists():
            return path

    return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string if longer than max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def validate_and_create_output_path(output_path: Union[str, Path], is_directory: bool = False) -> Path:
    """
    Validate and create output path, ensuring parent directories exist.

    Args:
        output_path: Output path
        is_directory: Whether path should be a directory

    Returns:
        Validated Path object
    """
    path = Path(output_path)

    if is_directory:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def get_device(device_spec: str = "auto") -> torch.device:
    """
    Get PyTorch device based on specification.

    Args:
        device_spec: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)

    Returns:
        PyTorch device object
    """
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_spec)


def chunks(lst: List, chunk_size: int):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]