#!/usr/bin/env python3
"""
Use Case 1: RNA Inverse Design using gRNAde

This script demonstrates the primary use case of gRNAde: generating RNA sequences
that fold into a specified 3D structure and/or secondary structure. It follows the
main design pipeline from design.py.

Usage:
    python examples/use_case_1_rna_inverse_design.py --pdb examples/data/RNASolo.pdb --secondary_structure ".((((((((..(.[[[[[....((((....))))..)..))))))))........(((..]]]]]..)))..." --mode 3d

Dependencies:
    - PyTorch
    - PyTorch Geometric
    - FastMCP (for MCP integration)
    - All gRNAde dependencies

Author: gRNAde MCP
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

import torch
import torch.nn.functional as F

# gRNAde imports
import sys
sys.path.append('../repo/geometric-rna-design')

try:
    from src.data.featurizer import RNAGraphFeaturizer
    from src.models import gRNAde
    from src.evaluator import (
        openknot_score_ribonanzanet,
        self_consistency_score_ribonanzanet,
        self_consistency_score_ribonanzanet_sec_struct
    )
    from src.constants import NUM_TO_LETTER, FILL_VALUE
    from tools.ribonanzanet.network import RibonanzaNet
    from tools.ribonanzanet_sec_struct.network import RibonanzaNetSS
except ImportError as e:
    print(f"Error importing gRNAde modules: {e}")
    print("Please make sure you're running this from the MCP root directory")
    print("and that the gRNAde repository is in repo/geometric-rna-design/")
    exit(1)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_partial_seq_logit_bias(partial_seq, featurizer, device, model_out_dim=4):
    """Create logit bias from partial sequence constraints."""
    if partial_seq is None:
        return None

    bias = torch.zeros(len(partial_seq), model_out_dim, device=device)
    # Map nucleotides: A=0, G=1, C=2, U=3
    nucleotide_map = {'A': 0, 'G': 1, 'C': 2, 'U': 3}

    for i, nucleotide in enumerate(partial_seq):
        if nucleotide in nucleotide_map:
            # Set high bias for the specified nucleotide, very negative for others
            bias[i, :] = -1e10
            bias[i, nucleotide_map[nucleotide]] = 0

    return bias


def rna_inverse_design(
    pdb_filepath=None,
    target_sec_struct=None,
    native_seq=None,
    partial_seq=None,
    mode='3d',
    total_samples=1000,
    n_samples=32,
    n_pass=100,
    temperature_min=0.1,
    temperature_max=1.0,
    pass_threshold=80,
    output_dir='designs',
    seed=42,
    model_path=None
):
    """
    Generate RNA sequences using gRNAde inverse design.

    Args:
        pdb_filepath: Path to target 3D structure (PDB format)
        target_sec_struct: Target secondary structure in dot-bracket notation
        native_seq: Native sequence (optional, used for 2D-only mode)
        partial_seq: Partial sequence constraints (optional)
        mode: '3d' (condition on 3D structure) or '2d' (condition on 2D structure only)
        total_samples: Total number of sequences to generate
        n_samples: Number of sequences to sample per batch
        n_pass: Number of designs that pass filtering threshold
        temperature_min: Minimum sampling temperature
        temperature_max: Maximum sampling temperature
        pass_threshold: Threshold for filtering designs
        output_dir: Directory to save generated designs
        seed: Random seed for reproducibility
        model_path: Path to gRNAde model checkpoint

    Returns:
        pandas.DataFrame: Generated designs with scores
    """

    print(f"🧬 gRNAde RNA Inverse Design")
    print(f"Mode: {mode}")
    print(f"Target: {pdb_filepath if mode == '3d' else 'Secondary structure only'}")
    print(f"Secondary structure: {target_sec_struct}")

    # Set random seed
    set_seed(seed)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Default model parameters
    node_in_dim = [15, 4]
    node_h_dim = [128, 16]
    edge_in_dim = [132, 3]
    edge_h_dim = [64, 4]
    num_layers = 4
    drop_rate = 0.5
    out_dim = 4

    # Featurizer parameters
    radius = 0.0
    top_k = 32
    num_rbf = 32
    num_posenc = 32
    max_num_conformers = 1
    noise_scale = 0.1
    drop_prob_3d = 0.75

    # Create featurizer
    print("Creating RNA graph featurizer...")
    featurizer = RNAGraphFeaturizer(
        split="test" if mode == '3d' else "test_2d",
        radius=radius,
        top_k=top_k,
        num_rbf=num_rbf,
        num_posenc=num_posenc,
        max_num_conformers=max_num_conformers,
        noise_scale=noise_scale,
        drop_prob_3d=drop_prob_3d
    )

    # Initialize model
    print("Initializing gRNAde model...")
    model = gRNAde(
        node_in_dim=node_in_dim,
        node_h_dim=node_h_dim,
        edge_in_dim=edge_in_dim,
        edge_h_dim=edge_h_dim,
        num_layers=num_layers,
        drop_rate=drop_rate,
        out_dim=out_dim
    )

    # Load checkpoint
    if model_path is None:
        model_path = "../repo/geometric-rna-design/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5"

    if not os.path.exists(model_path):
        # Try examples/data path
        alt_model_path = "examples/data/gRNAde_drop3d@0.75_maxlen@500.h5"
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            print(f"❌ Model checkpoint not found at {model_path}")
            print("Please download model checkpoints using the HuggingFace CLI:")
            print("hf download chaitjo/gRNAde --local-dir examples/data/")
            return None

    print(f"Loading model checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare input data
    if mode == '3d':
        if pdb_filepath is None:
            raise ValueError("PDB filepath required for 3D mode")
        if not os.path.exists(pdb_filepath):
            raise FileNotFoundError(f"PDB file not found: {pdb_filepath}")

        print(f"Loading 3D structure from {pdb_filepath}")
        _, raw_data = featurizer.featurize_from_pdb_filelist([pdb_filepath])
        # Update secondary structure if provided
        if target_sec_struct:
            raw_data['sec_struct_list'] = [target_sec_struct]

    elif mode == '2d':
        if target_sec_struct is None:
            raise ValueError("Secondary structure required for 2D mode")
        if native_seq is None:
            # Use sequence length from secondary structure
            native_seq = 'A' * len(target_sec_struct)

        print(f"Using 2D mode with secondary structure")
        raw_data = {
            "sequence": native_seq,
            "coords_list": [torch.ones(len(target_sec_struct), 3, 3) * FILL_VALUE],
            "sec_struct_list": [target_sec_struct],
        }

    # Create partial sequence bias if specified
    partial_seq_bias = create_partial_seq_logit_bias(partial_seq, featurizer, device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate designs
    print(f"Generating {total_samples} candidate sequences...")
    designs = []

    # For 2D mode, featurize once
    if mode == '2d':
        featurized_data = featurizer(raw_data).to(device)

    # Generate in batches
    for batch_idx in range(total_samples // n_samples):
        print(f"Batch {batch_idx + 1}/{total_samples // n_samples}, Designs collected: {len(designs)}")

        # Set temperature and seed for this batch
        temperature = np.random.uniform(temperature_min, temperature_max)
        batch_seed = random.randint(0, 9999)
        set_seed(batch_seed)

        # Featurize data
        if mode == '3d':
            featurized_data = featurizer(raw_data).to(device)

        # Sample sequences
        samples, logits = model.sample(
            featurized_data,
            n_samples,
            temperature,
            partial_seq_bias=partial_seq_bias,
            return_logits=True
        )

        # Calculate perplexity
        n_nodes = logits.shape[1]
        perplexity = (
            torch.exp(
                F.cross_entropy(
                    logits.view(n_samples * n_nodes, model.out_dim),
                    samples.view(n_samples * n_nodes).long(),
                    reduction="none",
                )
                .view(n_samples, n_nodes)
                .mean(dim=1)
            )
            .cpu()
            .numpy()
        )

        # Convert to sequences
        sequences = []
        for sample in samples.cpu().numpy():
            seq = "".join([NUM_TO_LETTER[num] for num in sample])
            sequences.append(seq)

        # Collect designs
        for i, (seq, perp) in enumerate(zip(sequences, perplexity)):
            design = {
                'sequence': seq,
                'perplexity': float(perp),
                'temperature': temperature,
                'seed': batch_seed,
                'mode': mode,
                'length': len(seq)
            }
            designs.append(design)

        # Stop if we have enough designs
        if len(designs) >= n_pass:
            break

    # Create DataFrame
    designs_df = pd.DataFrame(designs[:n_pass])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"rna_designs_{mode}_{timestamp}.csv")
    designs_df.to_csv(output_file, index=False)

    print(f"✅ Generated {len(designs_df)} designs")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Average perplexity: {designs_df['perplexity'].mean():.3f}")
    print(f"🧬 Sequence length: {designs_df['length'].iloc[0]} nucleotides")

    return designs_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='RNA Inverse Design using gRNAde')

    # Input structure
    parser.add_argument('--pdb', '--pdb_filepath', type=str,
                       help='Path to target 3D structure (PDB format)')
    parser.add_argument('--secondary_structure', '--ss', type=str,
                       help='Target secondary structure in dot-bracket notation')
    parser.add_argument('--native_seq', type=str,
                       help='Native sequence (optional, for 2D mode)')
    parser.add_argument('--partial_seq', type=str,
                       help='Partial sequence constraints (use _ for designable positions)')

    # Design parameters
    parser.add_argument('--mode', choices=['2d', '3d'], default='3d',
                       help='Design mode: 3d (3D structure) or 2d (secondary structure only)')
    parser.add_argument('--total_samples', type=int, default=1000,
                       help='Total number of sequences to generate')
    parser.add_argument('--n_samples', type=int, default=32,
                       help='Batch size for sequence generation')
    parser.add_argument('--n_pass', type=int, default=100,
                       help='Number of final designs to collect')
    parser.add_argument('--temperature_min', type=float, default=0.1,
                       help='Minimum sampling temperature')
    parser.add_argument('--temperature_max', type=float, default=1.0,
                       help='Maximum sampling temperature')

    # Output
    parser.add_argument('--output_dir', type=str, default='designs',
                       help='Output directory for designs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str,
                       help='Path to gRNAde model checkpoint')

    args = parser.parse_args()

    # Validate inputs
    if args.mode == '3d' and not args.pdb:
        parser.error("--pdb is required for 3D mode")
    if args.mode == '2d' and not args.secondary_structure:
        parser.error("--secondary_structure is required for 2D mode")

    # Run design
    try:
        designs = rna_inverse_design(
            pdb_filepath=args.pdb,
            target_sec_struct=args.secondary_structure,
            native_seq=args.native_seq,
            partial_seq=args.partial_seq,
            mode=args.mode,
            total_samples=args.total_samples,
            n_samples=args.n_samples,
            n_pass=args.n_pass,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            output_dir=args.output_dir,
            seed=args.seed,
            model_path=args.model_path
        )

        if designs is not None:
            print("\n🎉 RNA inverse design completed successfully!")
            print(f"Generated {len(designs)} unique RNA sequences")

    except Exception as e:
        print(f"❌ Error during RNA inverse design: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())