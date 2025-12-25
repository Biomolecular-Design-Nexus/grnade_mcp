#!/usr/bin/env python3
"""
Use Case 2: RNA Design Evaluation and Scoring

This script evaluates RNA sequences using multiple computational metrics:
- OpenKnot Score (SHAPE-to-secondary structure match)
- Self-consistency scores using RibonanzaNet
- Secondary structure prediction accuracy

Usage:
    python examples/use_case_2_rna_evaluation.py --sequences_file sequences.csv --target_structure "((...))" --output evaluation_results.csv

Dependencies:
    - PyTorch
    - RibonanzaNet models
    - All gRNAde dependencies

Author: gRNAde MCP
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

import torch

# gRNAde imports
import sys
sys.path.append('../repo/geometric-rna-design')

try:
    from src.evaluator import (
        openknot_score_ribonanzanet,
        self_consistency_score_ribonanzanet,
        self_consistency_score_ribonanzanet_sec_struct
    )
    from src.constants import LETTER_TO_NUM
    from tools.ribonanzanet.network import RibonanzaNet
    from tools.ribonanzanet_sec_struct.network import RibonanzaNetSS
except ImportError as e:
    print(f"Error importing gRNAde modules: {e}")
    print("Please make sure you're running this from the MCP root directory")
    exit(1)


def sequences_to_indices(sequences):
    """Convert RNA sequences to numerical indices."""
    indices = []
    for seq in sequences:
        seq_indices = [LETTER_TO_NUM[nuc] for nuc in seq.upper()]
        indices.append(seq_indices)
    return np.array(indices)


def evaluate_rna_sequences(
    sequences,
    target_sec_struct,
    ribonanza_model_path=None,
    ribonanza_ss_model_path=None,
    device='auto'
):
    """
    Evaluate RNA sequences using computational metrics.

    Args:
        sequences: List of RNA sequences to evaluate
        target_sec_struct: Target secondary structure in dot-bracket notation
        ribonanza_model_path: Path to RibonanzaNet model
        ribonanza_ss_model_path: Path to RibonanzaNetSS model
        device: Device for computation ('auto', 'cpu', 'cuda')

    Returns:
        pandas.DataFrame: Evaluation results for each sequence
    """

    print(f"🔬 Evaluating {len(sequences)} RNA sequences")
    print(f"Target structure: {target_sec_struct}")

    # Set device
    if device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load models
    print("Loading evaluation models...")

    # RibonanzaNet for SHAPE reactivity prediction
    if ribonanza_model_path is None:
        ribonanza_model_path = "../repo/geometric-rna-design/checkpoints/ribonanzanet/ribonanzanet.pt"

    if not os.path.exists(ribonanza_model_path):
        alt_path = "examples/data/ribonanzanet.pt"
        if os.path.exists(alt_path):
            ribonanza_model_path = alt_path
        else:
            print(f"❌ RibonanzaNet model not found at {ribonanza_model_path}")
            print("Please download checkpoints using: hf download chaitjo/gRNAde --local-dir examples/data/")
            return None

    ribonanza_net = RibonanzaNet(
        "../repo/geometric-rna-design/tools/ribonanzanet/config.yaml",
        ribonanza_model_path,
        device
    )
    ribonanza_net = ribonanza_net.to(device)
    ribonanza_net.eval()

    # RibonanzaNetSS for secondary structure prediction
    if ribonanza_ss_model_path is None:
        ribonanza_ss_model_path = "../repo/geometric-rna-design/checkpoints/ribonanzanet_sec_struct/ribonanzanet_ss.pt"

    if not os.path.exists(ribonanza_ss_model_path):
        alt_path = "examples/data/ribonanzanet_ss.pt"
        if os.path.exists(alt_path):
            ribonanza_ss_model_path = alt_path
        else:
            print(f"❌ RibonanzaNetSS model not found at {ribonanza_ss_model_path}")
            print("Please download checkpoints using: hf download chaitjo/gRNAde --local-dir examples/data/")
            return None

    ribonanza_net_ss = RibonanzaNetSS(
        "../repo/geometric-rna-design/tools/ribonanzanet_sec_struct/config.yaml",
        ribonanza_ss_model_path,
        device
    )
    ribonanza_net_ss = ribonanza_net_ss.to(device)
    ribonanza_net_ss.eval()

    # Convert sequences to indices
    seq_indices = sequences_to_indices(sequences)
    mask_seq = np.ones(seq_indices.shape, dtype=bool)

    # Evaluate sequences
    print("Computing evaluation metrics...")

    results = []

    # Batch evaluation for efficiency
    batch_size = min(32, len(sequences))
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sequences))

        batch_sequences = sequences[start_idx:end_idx]
        batch_seq_indices = seq_indices[start_idx:end_idx]
        batch_mask = mask_seq[start_idx:end_idx]

        print(f"Processing batch {batch_idx + 1}/{num_batches}")

        # OpenKnot Score (SHAPE-to-secondary structure match)
        try:
            openknot_scores = openknot_score_ribonanzanet(
                batch_seq_indices,
                target_sec_struct,
                batch_mask,
                ribonanza_net
            )
        except Exception as e:
            print(f"Warning: OpenKnot scoring failed: {e}")
            openknot_scores = np.zeros(len(batch_sequences))

        # Self-consistency score (SHAPE reactivity)
        try:
            sc_scores_ribonanza = self_consistency_score_ribonanzanet(
                batch_seq_indices,
                target_sec_struct,
                batch_mask,
                ribonanza_net
            )
        except Exception as e:
            print(f"Warning: SHAPE self-consistency scoring failed: {e}")
            sc_scores_ribonanza = np.zeros(len(batch_sequences))

        # Self-consistency score (secondary structure)
        try:
            sc_scores_ss = self_consistency_score_ribonanzanet_sec_struct(
                batch_seq_indices,
                target_sec_struct,
                batch_mask,
                ribonanza_net_ss
            )
        except Exception as e:
            print(f"Warning: Structure self-consistency scoring failed: {e}")
            sc_scores_ss = np.zeros(len(batch_sequences))

        # Collect results for this batch
        for i, seq in enumerate(batch_sequences):
            result = {
                'sequence': seq,
                'length': len(seq),
                'gc_content': (seq.count('G') + seq.count('C')) / len(seq),
                'openknot_score': float(openknot_scores[i]),
                'sc_score_ribonanzanet': float(sc_scores_ribonanza[i]),
                'sc_score_ribonanzanet_ss': float(sc_scores_ss[i])
            }
            results.append(result)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    print(f"✅ Evaluation completed")
    print(f"📊 Average OpenKnot Score: {results_df['openknot_score'].mean():.2f}")
    print(f"📊 Average SHAPE self-consistency: {results_df['sc_score_ribonanzanet'].mean():.4f}")
    print(f"📊 Average structure self-consistency: {results_df['sc_score_ribonanzanet_ss'].mean():.4f}")

    return results_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='RNA Design Evaluation and Scoring')

    # Input
    parser.add_argument('--sequences_file', '-f', type=str, required=True,
                       help='CSV file containing RNA sequences (must have "sequence" column)')
    parser.add_argument('--sequences', '-s', nargs='+',
                       help='Individual RNA sequences to evaluate')
    parser.add_argument('--target_structure', '--ss', type=str, required=True,
                       help='Target secondary structure in dot-bracket notation')

    # Model paths
    parser.add_argument('--ribonanza_model', type=str,
                       help='Path to RibonanzaNet model checkpoint')
    parser.add_argument('--ribonanza_ss_model', type=str,
                       help='Path to RibonanzaNetSS model checkpoint')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output CSV file for results')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device for computation')

    args = parser.parse_args()

    # Load sequences
    sequences = []
    if args.sequences_file:
        if not os.path.exists(args.sequences_file):
            print(f"❌ Sequences file not found: {args.sequences_file}")
            return 1

        print(f"Loading sequences from {args.sequences_file}")
        df = pd.read_csv(args.sequences_file)

        if 'sequence' not in df.columns:
            print("❌ CSV file must contain a 'sequence' column")
            return 1

        sequences.extend(df['sequence'].tolist())

    if args.sequences:
        sequences.extend(args.sequences)

    if not sequences:
        print("❌ No sequences provided")
        return 1

    # Remove duplicates while preserving order
    seen = set()
    unique_sequences = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            unique_sequences.append(seq)

    print(f"Loaded {len(unique_sequences)} unique sequences")

    # Run evaluation
    try:
        results = evaluate_rna_sequences(
            sequences=unique_sequences,
            target_sec_struct=args.target_structure,
            ribonanza_model_path=args.ribonanza_model,
            ribonanza_ss_model_path=args.ribonanza_ss_model,
            device=args.device
        )

        if results is None:
            return 1

        # Save results
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rna_evaluation_{timestamp}.csv"

        results.to_csv(output_file, index=False)
        print(f"📁 Results saved to: {output_file}")

        # Display summary statistics
        print("\n📈 Evaluation Summary:")
        print("=" * 50)
        print(f"Number of sequences: {len(results)}")
        print(f"Average sequence length: {results['length'].mean():.1f}")
        print(f"Average GC content: {results['gc_content'].mean():.2%}")
        print(f"OpenKnot Score: {results['openknot_score'].mean():.2f} ± {results['openknot_score'].std():.2f}")
        print(f"SHAPE self-consistency: {results['sc_score_ribonanzanet'].mean():.4f} ± {results['sc_score_ribonanzanet'].std():.4f}")
        print(f"Structure self-consistency: {results['sc_score_ribonanzanet_ss'].mean():.4f} ± {results['sc_score_ribonanzanet_ss'].std():.4f}")

        # Top-performing sequences
        print("\n🏆 Top 5 sequences by OpenKnot Score:")
        top_sequences = results.nlargest(5, 'openknot_score')
        for i, row in top_sequences.iterrows():
            print(f"{row['sequence']} (Score: {row['openknot_score']:.2f})")

        print("\n🎉 RNA evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Error during RNA evaluation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())