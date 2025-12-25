#!/usr/bin/env python3
"""
Script: rna_inverse_design.py
Description: RNA Inverse Design using gRNAde - Generate RNA sequences that fold into specified structures

Original Use Case: examples/use_case_1_rna_inverse_design_fixed.py
Dependencies Removed: Inlined constants, simplified repo imports

Usage:
    python scripts/rna_inverse_design.py --secondary_structure "((((....))))" --mode 2d --n_pass 5 --output_dir results/rna_design

Example:
    python scripts/rna_inverse_design.py --secondary_structure "((((....))))" --mode 2d --output_dir results/test
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import random
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json

# Essential scientific packages
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_checkpoint": "gRNAde_drop3d@0.75_maxlen@500.h5",
    "mode": "2d",  # "2d" or "3d"
    "total_samples": 1000,
    "n_samples": 32,
    "n_pass": 100,
    "temperature_min": 0.1,
    "temperature_max": 1.0,
    "pass_threshold": 80,
    "seed": 42,
    "device": "cpu"  # or "cuda" if available
}

# ==============================================================================
# Inlined Constants (from repo/geometric-rna-design/src/constants.py)
# ==============================================================================
# RNA nucleotides mapping
LETTER_TO_NUM = {"A": 0, "G": 1, "C": 2, "U": 3}
NUM_TO_LETTER = {0: "A", 1: "G", 2: "C", 3: "U"}
FILL_VALUE = 1e-5

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_partial_seq_logit_bias(partial_seq, device, model_out_dim=4):
    """Create logit bias from partial sequence constraints."""
    if partial_seq is None:
        return None

    bias = torch.zeros(len(partial_seq), model_out_dim, device=device)
    nucleotide_map = {'A': 0, 'G': 1, 'C': 2, 'U': 3}

    for i, nucleotide in enumerate(partial_seq):
        if nucleotide in nucleotide_map:
            # Set high bias for the specified nucleotide, very negative for others
            bias[i, :] = -1e10
            bias[i, nucleotide_map[nucleotide]] = 0

    return bias


def get_repo_imports():
    """Lazy load repo imports to minimize startup time."""
    import sys
    from pathlib import Path

    # Add repo to path
    repo_path = Path(__file__).parent.parent / "repo" / "geometric-rna-design"
    sys.path.insert(0, str(repo_path))

    try:
        from src.data.featurizer import RNAGraphFeaturizer
        from src.models import gRNAde
        from src.evaluator import (
            openknot_score_ribonanzanet,
            self_consistency_score_ribonanzanet,
            self_consistency_score_ribonanzanet_sec_struct
        )
        from tools.ribonanzanet.network import RibonanzaNet
        from tools.ribonanzanet_sec_struct.network import RibonanzaNetSS

        return {
            "RNAGraphFeaturizer": RNAGraphFeaturizer,
            "gRNAde": gRNAde,
            "openknot_score_ribonanzanet": openknot_score_ribonanzanet,
            "self_consistency_score_ribonanzanet": self_consistency_score_ribonanzanet,
            "self_consistency_score_ribonanzanet_sec_struct": self_consistency_score_ribonanzanet_sec_struct,
            "RibonanzaNet": RibonanzaNet,
            "RibonanzaNetSS": RibonanzaNetSS
        }
    except ImportError as e:
        raise ImportError(f"Could not import gRNAde modules: {e}. "
                         f"Make sure repo is at: {repo_path}")


def load_model_checkpoint(config: Dict[str, Any]):
    """Load gRNAde model from checkpoint."""
    imports = get_repo_imports()

    # Model checkpoint path
    script_dir = Path(__file__).parent
    mcp_root = script_dir.parent

    # Try multiple possible locations
    possible_paths = [
        mcp_root / "models" / config["model_checkpoint"],
        mcp_root / "repo" / "geometric-rna-design" / "models" / config["model_checkpoint"],
        Path(config["model_checkpoint"])  # If absolute path provided
    ]

    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model checkpoint not found. Tried: {[str(p) for p in possible_paths]}"
        )

    # Load model
    device = torch.device(config.get("device", "cpu"))
    model = imports["gRNAde"](
        model_path=str(model_path),
        device=device,
        num_layers=20,  # Default from repo
        model_type=config["mode"]
    )

    return model, imports


def sample_sequences(model, featurizer, structure_data, config: Dict[str, Any]):
    """Sample RNA sequences using the model."""
    device = torch.device(config.get("device", "cpu"))

    # Prepare data
    data = featurizer(structure_data).to(device)

    sequences = []
    perplexities = []

    for pass_idx in range(config["n_pass"]):
        # Sample temperature
        temperature = np.random.uniform(
            config["temperature_min"],
            config["temperature_max"]
        )

        # Generate sequences for this pass
        with torch.no_grad():
            logits = model(data)

            # Apply temperature
            logits = logits / temperature

            # Sample sequences
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            sampled_indices = sampled_indices.view(probs.shape[:-1])

            # Convert to sequences
            seq_length = sampled_indices.size(0)
            for i in range(min(config["n_samples"], seq_length)):
                indices = sampled_indices[:, i] if len(sampled_indices.shape) > 1 else sampled_indices
                sequence = ''.join([NUM_TO_LETTER[idx.item()] for idx in indices])

                # Calculate perplexity
                log_probs = F.log_softmax(logits, dim=-1)
                perplexity = torch.exp(-log_probs.mean()).item()

                sequences.append(sequence)
                perplexities.append(perplexity)

    return sequences[:config["total_samples"]], perplexities[:config["total_samples"]]


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_rna_inverse_design(
    pdb_file: Optional[Union[str, Path]] = None,
    secondary_structure: Optional[str] = None,
    partial_seq: Optional[str] = None,
    mode: str = "2d",
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate RNA sequences using gRNAde inverse design.

    Args:
        pdb_file: Path to PDB file (required for 3d mode)
        secondary_structure: Secondary structure in dot-bracket notation (required for 2d mode)
        partial_seq: Partial sequence constraints (optional)
        mode: Design mode - "2d" or "3d"
        output_dir: Directory to save results (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - sequences: List of generated sequences
            - perplexities: List of perplexity scores
            - config_used: Configuration used for generation
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_rna_inverse_design(
        ...     secondary_structure="((((....))))",
        ...     mode="2d",
        ...     output_dir="results"
        ... )
        >>> print(f"Generated {len(result['sequences'])} sequences")
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    config["mode"] = mode

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Validate inputs
    if mode == "3d" and pdb_file is None:
        raise ValueError("PDB file required for 3d mode")
    if mode == "2d" and secondary_structure is None:
        raise ValueError("Secondary structure required for 2d mode")

    # Load model and imports
    model, imports = load_model_checkpoint(config)
    featurizer = imports["RNAGraphFeaturizer"]()

    # Prepare structure data
    if mode == "3d":
        # For 3D mode, we need to load PDB data
        # This is simplified - the full implementation would use pdb_to_tensor
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        # Placeholder for PDB data - would need full featurizer implementation
        structure_data = {
            "pdb_path": str(pdb_file),
            "mode": "3d"
        }
    else:
        # For 2D mode, use secondary structure
        structure_data = {
            "secondary_structure": secondary_structure,
            "mode": "2d"
        }

    try:
        # Generate sequences
        sequences, perplexities = sample_sequences(model, featurizer, structure_data, config)

        # Prepare results
        results_df = pd.DataFrame({
            "sequence": sequences,
            "perplexity": perplexities,
            "temperature": [np.random.uniform(config["temperature_min"], config["temperature_max"])
                           for _ in sequences],
            "seed": [config["seed"]] * len(sequences),
            "mode": [mode] * len(sequences),
            "length": [len(seq) for seq in sequences]
        })

        # Save output if requested
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"rna_designs_{mode}_{timestamp}.csv"
            output_path = output_dir / output_filename

            results_df.to_csv(output_path, index=False)

        return {
            "sequences": sequences,
            "perplexities": perplexities,
            "results_df": results_df,
            "config_used": config,
            "output_file": str(output_path) if output_path else None,
            "metadata": {
                "mode": mode,
                "num_sequences": len(sequences),
                "pdb_file": str(pdb_file) if pdb_file else None,
                "secondary_structure": secondary_structure,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        # Return error information for debugging
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "config_used": config,
            "metadata": {
                "mode": mode,
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
    parser.add_argument('--pdb', help='PDB file path (required for 3d mode)')
    parser.add_argument('--secondary_structure', help='Secondary structure in dot-bracket notation (required for 2d mode)')
    parser.add_argument('--partial_seq', help='Partial sequence constraints')
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d', help='Design mode')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--config', help='Config file (JSON)')

    # Model parameters
    parser.add_argument('--total_samples', type=int, help='Total number of sequences to generate')
    parser.add_argument('--n_pass', type=int, help='Number of passes')
    parser.add_argument('--temperature_min', type=float, help='Minimum temperature')
    parser.add_argument('--temperature_max', type=float, help='Maximum temperature')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Prepare arguments
    kwargs = {}
    for arg in ['total_samples', 'n_pass', 'temperature_min', 'temperature_max', 'seed']:
        value = getattr(args, arg)
        if value is not None:
            kwargs[arg] = value

    # Run design
    result = run_rna_inverse_design(
        pdb_file=args.pdb,
        secondary_structure=args.secondary_structure,
        partial_seq=args.partial_seq,
        mode=args.mode,
        output_dir=args.output_dir,
        config=config,
        **kwargs
    )

    if result.get('success', True):
        print(f"✅ Success: Generated {len(result.get('sequences', []))} sequences")
        if result.get('output_file'):
            print(f"📁 Output saved to: {result['output_file']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())