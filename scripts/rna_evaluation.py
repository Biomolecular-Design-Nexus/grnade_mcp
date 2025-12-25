#!/usr/bin/env python3
"""
Script: rna_evaluation.py
Description: Evaluate RNA sequences using computational metrics (OpenKnot, RibonanzaNet self-consistency)

Original Use Case: examples/use_case_2_rna_evaluation_fixed.py
Dependencies Removed: Inlined constants, simplified evaluation functions

Usage:
    python scripts/rna_evaluation.py --sequences_file sequences.csv --target_structure "((...))" --output evaluation_results.csv

Example:
    python scripts/rna_evaluation.py --sequences_file examples/data/sequences/evaluation_test_sequences.csv --target_structure "((((....))))" --output results/evaluation.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# Essential scientific packages
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "ribonanza_model": "ribonanzanet.pt",
    "ribonanza_ss_model": "ribonanzanet_ss.pt",
    "device": "auto",  # "auto", "cpu", or "cuda"
    "batch_size": 32,
    "include_basic_stats": True,
    "verbose": True
}

# ==============================================================================
# Inlined Constants (from repo/geometric-rna-design/src/constants.py)
# ==============================================================================
# RNA nucleotides mapping
LETTER_TO_NUM = {"A": 0, "G": 1, "C": 2, "U": 3}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def sequences_to_indices(sequences: List[str]) -> np.ndarray:
    """Convert RNA sequences to numerical indices."""
    indices = []
    for seq in sequences:
        seq_indices = [LETTER_TO_NUM[nuc] for nuc in seq.upper() if nuc in LETTER_TO_NUM]
        indices.append(seq_indices)
    return indices


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of an RNA sequence."""
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0


def get_repo_imports():
    """Lazy load repo imports to minimize startup time."""
    import sys
    from pathlib import Path

    # Add repo to path
    repo_path = Path(__file__).parent.parent / "repo" / "geometric-rna-design"
    sys.path.insert(0, str(repo_path))

    try:
        from src.evaluator import (
            openknot_score_ribonanzanet,
            self_consistency_score_ribonanzanet,
            self_consistency_score_ribonanzanet_sec_struct
        )
        from tools.ribonanzanet.network import RibonanzaNet
        from tools.ribonanzanet_sec_struct.network import RibonanzaNetSS

        return {
            "openknot_score_ribonanzanet": openknot_score_ribonanzanet,
            "self_consistency_score_ribonanzanet": self_consistency_score_ribonanzanet,
            "self_consistency_score_ribonanzanet_sec_struct": self_consistency_score_ribonanzanet_sec_struct,
            "RibonanzaNet": RibonanzaNet,
            "RibonanzaNetSS": RibonanzaNetSS
        }
    except ImportError as e:
        raise ImportError(f"Could not import gRNAde modules: {e}. "
                         f"Make sure repo is at: {repo_path}")


def load_ribonanza_models(config: Dict[str, Any]):
    """Load RibonanzaNet models for evaluation."""
    imports = get_repo_imports()

    # Model paths
    script_dir = Path(__file__).parent
    mcp_root = script_dir.parent

    # Try multiple possible locations for models
    model_locations = [
        mcp_root / "models",
        mcp_root / "examples" / "data",
        mcp_root / "repo" / "geometric-rna-design" / "checkpoints" / "ribonanzanet"
    ]

    ribonanza_path = None
    ribonanza_ss_path = None

    for loc in model_locations:
        if (loc / config["ribonanza_model"]).exists():
            ribonanza_path = loc / config["ribonanza_model"]
        if (loc / config["ribonanza_ss_model"]).exists():
            ribonanza_ss_path = loc / config["ribonanza_ss_model"]

    if ribonanza_path is None:
        raise FileNotFoundError(f"RibonanzaNet model not found: {config['ribonanza_model']}")
    if ribonanza_ss_path is None:
        raise FileNotFoundError(f"RibonanzaNetSS model not found: {config['ribonanza_ss_model']}")

    # Set device
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])

    # Load models
    try:
        ribonanza_model = imports["RibonanzaNet"](device=device)
        ribonanza_model.load_state_dict(torch.load(ribonanza_path, map_location=device))
        ribonanza_model.eval()

        ribonanza_ss_model = imports["RibonanzaNetSS"](device=device)
        ribonanza_ss_model.load_state_dict(torch.load(ribonanza_ss_path, map_location=device))
        ribonanza_ss_model.eval()

        return ribonanza_model, ribonanza_ss_model, device

    except Exception as e:
        # If model loading fails, return None to use basic evaluation only
        print(f"⚠️  Warning: Could not load evaluation models: {e}")
        print("Will use basic sequence statistics only.")
        return None, None, device


def evaluate_sequence_basic(sequence: str) -> Dict[str, Any]:
    """Calculate basic sequence statistics."""
    return {
        "length": len(sequence),
        "gc_content": calculate_gc_content(sequence),
        "a_count": sequence.upper().count('A'),
        "g_count": sequence.upper().count('G'),
        "c_count": sequence.upper().count('C'),
        "u_count": sequence.upper().count('U')
    }


def evaluate_sequence_advanced(sequence: str, target_structure: str, models: tuple, imports: dict) -> Dict[str, Any]:
    """Evaluate sequence using advanced metrics (with error handling)."""
    ribonanza_model, ribonanza_ss_model, device = models

    if ribonanza_model is None or ribonanza_ss_model is None:
        return {
            "openknot_score": 0.0,
            "sc_score_ribonanzanet": 0.0,
            "sc_score_ribonanzanet_ss": 0.0
        }

    try:
        # Convert sequence to indices
        seq_indices = [LETTER_TO_NUM[nuc] for nuc in sequence.upper() if nuc in LETTER_TO_NUM]

        if len(seq_indices) == 0:
            return {
                "openknot_score": 0.0,
                "sc_score_ribonanzanet": 0.0,
                "sc_score_ribonanzanet_ss": 0.0
            }

        # Convert to tensor (simplified version)
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long).to(device)

        # OpenKnot score (with error handling)
        try:
            openknot_score = imports["openknot_score_ribonanzanet"](
                sequences=[sequence],
                target_sec_struct=target_structure,
                model=ribonanza_model,
                device=device
            )
            # Handle if it returns array
            if hasattr(openknot_score, '__len__') and len(openknot_score) > 0:
                openknot_score = float(openknot_score[0])
            else:
                openknot_score = float(openknot_score) if openknot_score is not None else 0.0
        except Exception as e:
            print(f"⚠️  OpenKnot scoring failed: {e}")
            openknot_score = 0.0

        # Self-consistency score (SHAPE) (with error handling)
        try:
            sc_ribonanza = imports["self_consistency_score_ribonanzanet"](
                sequences=[sequence],
                model=ribonanza_model,
                device=device
            )
            # Handle if it returns array
            if hasattr(sc_ribonanza, '__len__') and len(sc_ribonanza) > 0:
                sc_ribonanza = float(sc_ribonanza[0])
            else:
                sc_ribonanza = float(sc_ribonanza) if sc_ribonanza is not None else 0.0
        except Exception as e:
            print(f"⚠️  SHAPE SC scoring failed: {e}")
            sc_ribonanza = 0.0

        # Self-consistency score (SS) (with error handling)
        try:
            sc_ribonanza_ss = imports["self_consistency_score_ribonanzanet_sec_struct"](
                sequences=[sequence],
                model=ribonanza_ss_model,
                device=device
            )
            # Handle if it returns array
            if hasattr(sc_ribonanza_ss, '__len__') and len(sc_ribonanza_ss) > 0:
                sc_ribonanza_ss = float(sc_ribonanza_ss[0])
            else:
                sc_ribonanza_ss = float(sc_ribonanza_ss) if sc_ribonanza_ss is not None else 0.0
        except Exception as e:
            print(f"⚠️  Structure SC scoring failed: {e}")
            sc_ribonanza_ss = 0.0

        return {
            "openknot_score": openknot_score,
            "sc_score_ribonanzanet": sc_ribonanza,
            "sc_score_ribonanzanet_ss": sc_ribonanza_ss
        }

    except Exception as e:
        print(f"⚠️  Advanced evaluation failed for sequence {sequence[:10]}...: {e}")
        return {
            "openknot_score": 0.0,
            "sc_score_ribonanzanet": 0.0,
            "sc_score_ribonanzanet_ss": 0.0
        }


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_rna_evaluation(
    sequences: Union[List[str], str, Path],
    target_structure: str,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate RNA sequences using computational metrics.

    Args:
        sequences: List of RNA sequences, or path to CSV file containing sequences
        target_structure: Target secondary structure in dot-bracket notation
        output_file: Path to save results (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results_df: DataFrame with evaluation results
            - summary_stats: Summary statistics
            - config_used: Configuration used
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_rna_evaluation(
        ...     sequences=["GGGAAACCC", "AUCGAUCGAUC"],
        ...     target_structure="(((...)))",
        ...     output_file="results/evaluation.csv"
        ... )
        >>> print(result['results_df'])
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load sequences
    if isinstance(sequences, (str, Path)):
        # Load from file
        sequences_file = Path(sequences)
        if not sequences_file.exists():
            raise FileNotFoundError(f"Sequences file not found: {sequences_file}")

        df = pd.read_csv(sequences_file)
        if 'sequence' in df.columns:
            sequences_list = df['sequence'].tolist()
        elif 'seq' in df.columns:
            sequences_list = df['seq'].tolist()
        else:
            # Assume first column contains sequences
            sequences_list = df.iloc[:, 0].tolist()
    else:
        sequences_list = list(sequences)

    if config["verbose"]:
        print(f"🔬 Evaluating {len(sequences_list)} RNA sequences")
        print(f"Target structure: {target_structure}")

    # Load models and imports
    try:
        imports = get_repo_imports()
        models = load_ribonanza_models(config)
        advanced_evaluation = models[0] is not None

        if config["verbose"]:
            if advanced_evaluation:
                print("✅ Advanced evaluation models loaded")
            else:
                print("⚠️  Using basic evaluation only (models not available)")

    except Exception as e:
        if config["verbose"]:
            print(f"⚠️  Could not load evaluation models: {e}")
            print("Using basic sequence statistics only")
        imports = None
        models = (None, None, None)
        advanced_evaluation = False

    # Evaluate sequences
    results = []

    for i, sequence in enumerate(sequences_list):
        if config["verbose"] and i % 10 == 0:
            print(f"Processing sequence {i+1}/{len(sequences_list)}")

        # Basic evaluation
        basic_stats = evaluate_sequence_basic(sequence)

        # Advanced evaluation (if available)
        if advanced_evaluation and imports:
            advanced_stats = evaluate_sequence_advanced(sequence, target_structure, models, imports)
        else:
            advanced_stats = {
                "openknot_score": 0.0,
                "sc_score_ribonanzanet": 0.0,
                "sc_score_ribonanzanet_ss": 0.0
            }

        # Combine results
        result_row = {
            "sequence": sequence,
            **basic_stats,
            **advanced_stats
        }
        results.append(result_row)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    summary_stats = {
        "total_sequences": len(sequences_list),
        "mean_length": results_df["length"].mean(),
        "mean_gc_content": results_df["gc_content"].mean(),
        "mean_openknot_score": results_df["openknot_score"].mean(),
        "mean_sc_ribonanzanet": results_df["sc_score_ribonanzanet"].mean(),
        "mean_sc_ribonanzanet_ss": results_df["sc_score_ribonanzanet_ss"].mean(),
        "advanced_evaluation_used": advanced_evaluation
    }

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        if config["verbose"]:
            print(f"📁 Results saved to: {output_path}")

    return {
        "results_df": results_df,
        "summary_stats": summary_stats,
        "config_used": config,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "target_structure": target_structure,
            "num_sequences": len(sequences_list),
            "advanced_evaluation": advanced_evaluation,
            "timestamp": datetime.now().isoformat()
        }
    }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sequences', help='Comma-separated sequences or path to CSV file')
    parser.add_argument('--sequences_file', help='Path to CSV file containing sequences')
    parser.add_argument('--target_structure', required=True, help='Target secondary structure')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--config', help='Config file (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Determine sequences input
    if args.sequences_file:
        sequences_input = args.sequences_file
    elif args.sequences:
        if ',' in args.sequences:
            sequences_input = args.sequences.split(',')
        else:
            sequences_input = [args.sequences]
    else:
        parser.error("Must provide either --sequences or --sequences_file")

    # Add CLI options to config
    if config is None:
        config = {}
    if args.verbose:
        config['verbose'] = True

    # Run evaluation
    try:
        result = run_rna_evaluation(
            sequences=sequences_input,
            target_structure=args.target_structure,
            output_file=args.output,
            config=config
        )

        print(f"✅ Success: Evaluated {result['metadata']['num_sequences']} sequences")
        if result.get('output_file'):
            print(f"📁 Output saved to: {result['output_file']}")

        # Print summary
        stats = result['summary_stats']
        print(f"\n📊 Summary Statistics:")
        print(f"   Mean length: {stats['mean_length']:.1f}")
        print(f"   Mean GC content: {stats['mean_gc_content']:.3f}")
        if stats['advanced_evaluation_used']:
            print(f"   Mean OpenKnot score: {stats['mean_openknot_score']:.3f}")
            print(f"   Mean SC (RibonanzaNet): {stats['mean_sc_ribonanzanet']:.3f}")
            print(f"   Mean SC (RibonanzaNet SS): {stats['mean_sc_ribonanzanet_ss']:.3f}")
        else:
            print("   Advanced scores: Not available (models not loaded)")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())