#!/usr/bin/env python3
"""
Use Case 4: High-Throughput RNA Design Pipeline

This script provides a complete pipeline for batch RNA design and screening:
- Multi-target design (multiple PDB structures or secondary structures)
- Parallel design generation with different parameters
- Comprehensive scoring and filtering
- Automated design ranking and selection
- Output preparation for experimental validation

Usage:
    python examples/use_case_4_batch_design_pipeline.py --targets_file targets.csv --output_dir batch_results/
    python examples/use_case_4_batch_design_pipeline.py --pdb_dir structures/ --mode 3d --n_designs_per_target 1000

Dependencies:
    - PyTorch
    - All gRNAde dependencies
    - Multiprocessing support

Author: gRNAde MCP
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import random

# gRNAde imports
import sys
sys.path.append('../repo/geometric-rna-design')

try:
    from examples.use_case_1_rna_inverse_design import rna_inverse_design
    from examples.use_case_2_rna_evaluation import evaluate_rna_sequences
    from src.constants import NUM_TO_LETTER, LETTER_TO_NUM
except ImportError as e:
    print(f"Error importing gRNAde modules: {e}")
    print("Please make sure you're running this from the MCP root directory")
    print("and that use_case_1 and use_case_2 scripts are available")
    exit(1)


def load_targets_from_csv(csv_file: str) -> List[Dict]:
    """Load design targets from CSV file."""
    df = pd.read_csv(csv_file)

    required_columns = ['target_name']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain at least: {required_columns}")

    targets = []
    for _, row in df.iterrows():
        target = {
            'name': row['target_name'],
            'pdb_file': row.get('pdb_file', None),
            'secondary_structure': row.get('secondary_structure', None),
            'mode': row.get('mode', '3d' if row.get('pdb_file') else '2d'),
            'native_seq': row.get('native_seq', None),
            'partial_seq': row.get('partial_seq', None),
            'description': row.get('description', '')
        }

        # Validate target
        if target['mode'] == '3d' and not target['pdb_file']:
            print(f"⚠️  Warning: Target {target['name']} set to 3D mode but no PDB file provided")
            target['mode'] = '2d'

        if target['mode'] == '2d' and not target['secondary_structure']:
            raise ValueError(f"Target {target['name']}: 2D mode requires secondary_structure")

        targets.append(target)

    return targets


def load_targets_from_directory(pdb_dir: str, mode: str = '3d') -> List[Dict]:
    """Load design targets from directory of PDB files."""
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))

    if not pdb_files:
        raise ValueError(f"No PDB files found in directory: {pdb_dir}")

    targets = []
    for pdb_file in pdb_files:
        basename = os.path.splitext(os.path.basename(pdb_file))[0]
        target = {
            'name': basename,
            'pdb_file': pdb_file,
            'secondary_structure': None,
            'mode': mode,
            'native_seq': None,
            'partial_seq': None,
            'description': f'Design from {basename}.pdb'
        }
        targets.append(target)

    return targets


def design_single_target(
    target: Dict,
    design_params: Dict,
    output_dir: str,
    target_idx: int = 0
) -> Dict:
    """Design sequences for a single target."""
    print(f"🎯 Designing for target: {target['name']} ({target['mode']} mode)")

    try:
        # Create target-specific output directory
        target_output_dir = os.path.join(output_dir, f"target_{target_idx:03d}_{target['name']}")
        os.makedirs(target_output_dir, exist_ok=True)

        # Run inverse design
        designs = rna_inverse_design(
            pdb_filepath=target['pdb_file'],
            target_sec_struct=target['secondary_structure'],
            native_seq=target['native_seq'],
            partial_seq=target['partial_seq'],
            mode=target['mode'],
            total_samples=design_params['total_samples'],
            n_samples=design_params['batch_size'],
            n_pass=design_params['n_designs_per_target'],
            temperature_min=design_params['temperature_min'],
            temperature_max=design_params['temperature_max'],
            output_dir=target_output_dir,
            seed=design_params['seed'] + target_idx,
            model_path=design_params.get('model_path')
        )

        if designs is None or designs.empty:
            return {
                'target': target,
                'success': False,
                'error': 'No designs generated',
                'designs': None,
                'output_dir': target_output_dir
            }

        # Add target metadata to designs
        designs['target_name'] = target['name']
        designs['target_mode'] = target['mode']
        designs['target_description'] = target['description']

        # Save target-specific results
        designs_file = os.path.join(target_output_dir, f"designs_{target['name']}.csv")
        designs.to_csv(designs_file, index=False)

        # Save target metadata
        target_info = {
            'target': target,
            'design_params': design_params,
            'num_designs': len(designs),
            'designs_file': designs_file,
            'timestamp': datetime.now().isoformat()
        }

        info_file = os.path.join(target_output_dir, f"target_info_{target['name']}.json")
        with open(info_file, 'w') as f:
            json.dump(target_info, f, indent=2)

        return {
            'target': target,
            'success': True,
            'designs': designs,
            'output_dir': target_output_dir,
            'num_designs': len(designs),
            'info_file': info_file
        }

    except Exception as e:
        return {
            'target': target,
            'success': False,
            'error': str(e),
            'designs': None,
            'output_dir': None
        }


def evaluate_batch_designs(
    design_results: List[Dict],
    evaluation_params: Dict,
    output_dir: str
) -> Dict:
    """Evaluate all generated designs with comprehensive scoring."""
    print("🔬 Evaluating batch designs with comprehensive scoring...")

    # Collect all successful designs
    all_designs = []
    for result in design_results:
        if result['success'] and result['designs'] is not None:
            all_designs.append(result['designs'])

    if not all_designs:
        print("❌ No designs to evaluate")
        return None

    # Combine all designs
    combined_designs = pd.concat(all_designs, ignore_index=True)
    print(f"📊 Evaluating {len(combined_designs)} total designs")

    # Group by target for evaluation
    evaluation_results = {}

    for target_name, target_designs in combined_designs.groupby('target_name'):
        print(f"🎯 Evaluating {len(target_designs)} designs for {target_name}")

        # Get target secondary structure
        target_info = next(r['target'] for r in design_results if r['target']['name'] == target_name)
        target_ss = target_info['secondary_structure']

        if not target_ss:
            print(f"⚠️  Warning: No secondary structure for {target_name}, skipping evaluation")
            continue

        try:
            # Run comprehensive evaluation
            eval_results = evaluate_rna_sequences(
                sequences=target_designs['sequence'].tolist(),
                target_sec_struct=target_ss,
                device=evaluation_params.get('device', 'auto')
            )

            if eval_results is not None:
                # Merge evaluation results with design data
                eval_results['target_name'] = target_name
                eval_results['design_index'] = target_designs.index.tolist()

                # Merge with original design data
                merged_results = pd.merge(
                    target_designs.reset_index(),
                    eval_results,
                    left_index=True,
                    right_index=True,
                    suffixes=('', '_eval')
                )

                evaluation_results[target_name] = merged_results

                print(f"✅ Evaluated {len(eval_results)} designs for {target_name}")
                print(f"   📊 Avg OpenKnot Score: {eval_results['openknot_score'].mean():.3f}")
                print(f"   📊 Avg SHAPE SC: {eval_results['sc_score_ribonanzanet'].mean():.4f}")

        except Exception as e:
            print(f"❌ Evaluation failed for {target_name}: {e}")
            evaluation_results[target_name] = target_designs

    # Save individual target evaluation results
    eval_dir = os.path.join(output_dir, 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)

    for target_name, eval_df in evaluation_results.items():
        eval_file = os.path.join(eval_dir, f"evaluation_{target_name}.csv")
        eval_df.to_csv(eval_file, index=False)

    # Create combined evaluation summary
    if evaluation_results:
        all_evaluated = pd.concat(evaluation_results.values(), ignore_index=True)

        summary_file = os.path.join(eval_dir, "batch_evaluation_summary.csv")
        all_evaluated.to_csv(summary_file, index=False)

        print(f"📁 Saved evaluation results to: {eval_dir}")

    return evaluation_results


def rank_and_filter_designs(
    evaluation_results: Dict,
    filtering_params: Dict,
    output_dir: str
) -> Dict:
    """Rank and filter designs based on multiple criteria."""
    print("🏆 Ranking and filtering designs...")

    if not evaluation_results:
        print("❌ No evaluation results to filter")
        return None

    filter_dir = os.path.join(output_dir, 'filtered')
    os.makedirs(filter_dir, exist_ok=True)

    filtering_results = {}

    for target_name, eval_df in evaluation_results.items():
        print(f"🎯 Filtering designs for {target_name}")

        # Apply score thresholds if available
        filtered_df = eval_df.copy()

        # Filter by OpenKnot score if available
        if 'openknot_score' in filtered_df.columns:
            openknot_threshold = filtering_params.get('min_openknot_score', 0.0)
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['openknot_score'] >= openknot_threshold]
            print(f"   OpenKnot filter (>= {openknot_threshold}): {before_count} → {len(filtered_df)}")

        # Filter by SHAPE self-consistency if available
        if 'sc_score_ribonanzanet' in filtered_df.columns:
            shape_threshold = filtering_params.get('max_shape_sc_error', 1.0)
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['sc_score_ribonanzanet'] <= shape_threshold]
            print(f"   SHAPE SC filter (<= {shape_threshold}): {before_count} → {len(filtered_df)}")

        # Filter by perplexity if available
        if 'perplexity' in filtered_df.columns:
            perplexity_threshold = filtering_params.get('max_perplexity', 100.0)
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['perplexity'] <= perplexity_threshold]
            print(f"   Perplexity filter (<= {perplexity_threshold}): {before_count} → {len(filtered_df)}")

        # Filter by GC content
        if 'gc_content' in filtered_df.columns:
            gc_min = filtering_params.get('min_gc_content', 0.0)
            gc_max = filtering_params.get('max_gc_content', 1.0)
            before_count = len(filtered_df)
            filtered_df = filtered_df[
                (filtered_df['gc_content'] >= gc_min) &
                (filtered_df['gc_content'] <= gc_max)
            ]
            print(f"   GC content filter ({gc_min:.1%}-{gc_max:.1%}): {before_count} → {len(filtered_df)}")

        # Rank by composite score
        if len(filtered_df) > 0:
            # Calculate composite score (higher is better)
            score_components = []

            if 'openknot_score' in filtered_df.columns:
                # Higher OpenKnot score is better
                score_components.append(filtered_df['openknot_score'])

            if 'sc_score_ribonanzanet' in filtered_df.columns:
                # Lower SHAPE error is better (invert)
                max_shape_error = filtered_df['sc_score_ribonanzanet'].max()
                score_components.append(max_shape_error - filtered_df['sc_score_ribonanzanet'])

            if 'perplexity' in filtered_df.columns:
                # Lower perplexity is better (invert)
                max_perplexity = filtered_df['perplexity'].max()
                score_components.append(max_perplexity - filtered_df['perplexity'])

            if score_components:
                # Normalize and combine scores
                normalized_scores = []
                for component in score_components:
                    norm_component = (component - component.min()) / (component.max() - component.min() + 1e-10)
                    normalized_scores.append(norm_component)

                filtered_df['composite_score'] = sum(normalized_scores) / len(normalized_scores)
            else:
                # Use perplexity as fallback (lower is better)
                if 'perplexity' in filtered_df.columns:
                    filtered_df['composite_score'] = 1.0 / (filtered_df['perplexity'] + 1.0)
                else:
                    filtered_df['composite_score'] = 1.0

            # Rank by composite score
            filtered_df = filtered_df.sort_values('composite_score', ascending=False)

            # Select top designs
            top_n = filtering_params.get('top_n_per_target', 10)
            top_designs = filtered_df.head(top_n)

            # Save filtered and ranked results
            filter_file = os.path.join(filter_dir, f"filtered_{target_name}.csv")
            filtered_df.to_csv(filter_file, index=False)

            top_file = os.path.join(filter_dir, f"top_{top_n}_{target_name}.csv")
            top_designs.to_csv(top_file, index=False)

            filtering_results[target_name] = {
                'filtered': filtered_df,
                'top_designs': top_designs,
                'filter_file': filter_file,
                'top_file': top_file,
                'original_count': len(eval_df),
                'filtered_count': len(filtered_df),
                'top_count': len(top_designs)
            }

            print(f"✅ Filtered {target_name}: {len(eval_df)} → {len(filtered_df)} → {len(top_designs)} top designs")

        else:
            print(f"❌ No designs passed filters for {target_name}")
            filtering_results[target_name] = {
                'filtered': pd.DataFrame(),
                'top_designs': pd.DataFrame(),
                'original_count': len(eval_df),
                'filtered_count': 0,
                'top_count': 0
            }

    # Create batch summary
    summary_data = []
    for target_name, result in filtering_results.items():
        summary_data.append({
            'target_name': target_name,
            'original_designs': result['original_count'],
            'filtered_designs': result['filtered_count'],
            'top_designs': result['top_count'],
            'success_rate': result['filtered_count'] / result['original_count'] if result['original_count'] > 0 else 0
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(filter_dir, "batch_filtering_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    print(f"📁 Saved filtering results to: {filter_dir}")

    return filtering_results


def run_batch_design_pipeline(
    targets: List[Dict],
    design_params: Dict,
    evaluation_params: Dict,
    filtering_params: Dict,
    output_dir: str,
    max_workers: int = None
) -> Dict:
    """
    Run complete batch RNA design pipeline.

    Args:
        targets: List of design targets
        design_params: Parameters for design generation
        evaluation_params: Parameters for design evaluation
        filtering_params: Parameters for design filtering
        output_dir: Output directory for all results
        max_workers: Maximum number of parallel workers

    Returns:
        dict: Complete pipeline results
    """

    print(f"🚀 Starting batch RNA design pipeline for {len(targets)} targets")
    print(f"📁 Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save pipeline configuration
    config = {
        'targets': targets,
        'design_params': design_params,
        'evaluation_params': evaluation_params,
        'filtering_params': filtering_params,
        'timestamp': datetime.now().isoformat(),
        'num_targets': len(targets)
    }

    config_file = os.path.join(output_dir, 'pipeline_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Parallel design generation
    print("\n🎯 Phase 1: Design Generation")
    print("=" * 50)

    if max_workers is None:
        max_workers = min(len(targets), mp.cpu_count())

    print(f"Using {max_workers} parallel workers")

    design_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit design jobs
        future_to_target = {
            executor.submit(
                design_single_target,
                target,
                design_params,
                output_dir,
                idx
            ): (idx, target) for idx, target in enumerate(targets)
        }

        # Collect results as they complete
        for future in as_completed(future_to_target):
            target_idx, target = future_to_target[future]
            try:
                result = future.result()
                design_results.append(result)

                if result['success']:
                    print(f"✅ Completed target {target_idx}: {target['name']} ({result['num_designs']} designs)")
                else:
                    print(f"❌ Failed target {target_idx}: {target['name']} - {result['error']}")

            except Exception as e:
                print(f"❌ Exception for target {target_idx}: {target['name']} - {e}")
                design_results.append({
                    'target': target,
                    'success': False,
                    'error': str(e),
                    'designs': None
                })

    # Evaluation phase
    print("\n🔬 Phase 2: Design Evaluation")
    print("=" * 50)

    evaluation_results = evaluate_batch_designs(design_results, evaluation_params, output_dir)

    # Filtering and ranking phase
    print("\n🏆 Phase 3: Design Filtering and Ranking")
    print("=" * 50)

    filtering_results = rank_and_filter_designs(evaluation_results, filtering_params, output_dir)

    # Final summary
    print("\n📊 Pipeline Summary")
    print("=" * 50)

    successful_targets = sum(1 for r in design_results if r['success'])
    total_designs = sum(r['num_designs'] for r in design_results if r['success'])

    print(f"Targets processed: {successful_targets}/{len(targets)}")
    print(f"Total designs generated: {total_designs}")

    if evaluation_results:
        evaluated_targets = len(evaluation_results)
        total_evaluated = sum(len(df) for df in evaluation_results.values())
        print(f"Targets evaluated: {evaluated_targets}")
        print(f"Total designs evaluated: {total_evaluated}")

    if filtering_results:
        filtered_targets = len(filtering_results)
        total_filtered = sum(r['filtered_count'] for r in filtering_results.values())
        total_top = sum(r['top_count'] for r in filtering_results.values())
        print(f"Targets filtered: {filtered_targets}")
        print(f"Total designs after filtering: {total_filtered}")
        print(f"Total top designs selected: {total_top}")

    # Compile final results
    pipeline_results = {
        'config': config,
        'design_results': design_results,
        'evaluation_results': evaluation_results,
        'filtering_results': filtering_results,
        'summary': {
            'total_targets': len(targets),
            'successful_targets': successful_targets,
            'total_designs': total_designs,
            'total_evaluated': total_evaluated if evaluation_results else 0,
            'total_filtered': total_filtered if filtering_results else 0,
            'total_top': total_top if filtering_results else 0,
            'success_rate': successful_targets / len(targets),
            'pipeline_duration': None  # Could add timing
        }
    }

    # Save pipeline results
    results_file = os.path.join(output_dir, 'pipeline_results.json')
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    print(f"\n📁 Complete pipeline results saved to: {results_file}")
    print("🎉 Batch RNA design pipeline completed!")

    return pipeline_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='High-Throughput RNA Design Pipeline')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--targets_file', type=str,
                           help='CSV file with design targets')
    input_group.add_argument('--pdb_dir', type=str,
                           help='Directory containing PDB files')

    # Design parameters
    parser.add_argument('--mode', choices=['2d', '3d'], default='3d',
                       help='Default design mode (default: 3d)')
    parser.add_argument('--n_designs_per_target', type=int, default=100,
                       help='Number of designs to generate per target')
    parser.add_argument('--total_samples', type=int, default=10000,
                       help='Total samples to generate per target')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for design generation')
    parser.add_argument('--temperature_min', type=float, default=0.1,
                       help='Minimum sampling temperature')
    parser.add_argument('--temperature_max', type=float, default=1.0,
                       help='Maximum sampling temperature')

    # Filtering parameters
    parser.add_argument('--min_openknot_score', type=float, default=0.0,
                       help='Minimum OpenKnot score for filtering')
    parser.add_argument('--max_shape_sc_error', type=float, default=1.0,
                       help='Maximum SHAPE self-consistency error')
    parser.add_argument('--max_perplexity', type=float, default=100.0,
                       help='Maximum perplexity for filtering')
    parser.add_argument('--min_gc_content', type=float, default=0.0,
                       help='Minimum GC content (0-1)')
    parser.add_argument('--max_gc_content', type=float, default=1.0,
                       help='Maximum GC content (0-1)')
    parser.add_argument('--top_n_per_target', type=int, default=10,
                       help='Number of top designs to select per target')

    # Execution parameters
    parser.add_argument('--max_workers', type=int,
                       help='Maximum number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device for computation')

    # Output
    parser.add_argument('--output_dir', type=str, default='batch_design_results',
                       help='Output directory for all results')

    args = parser.parse_args()

    # Load targets
    if args.targets_file:
        if not os.path.exists(args.targets_file):
            print(f"❌ Targets file not found: {args.targets_file}")
            return 1
        targets = load_targets_from_csv(args.targets_file)
    else:
        if not os.path.isdir(args.pdb_dir):
            print(f"❌ PDB directory not found: {args.pdb_dir}")
            return 1
        targets = load_targets_from_directory(args.pdb_dir, args.mode)

    print(f"📋 Loaded {len(targets)} design targets")

    # Setup parameters
    design_params = {
        'total_samples': args.total_samples,
        'batch_size': args.batch_size,
        'n_designs_per_target': args.n_designs_per_target,
        'temperature_min': args.temperature_min,
        'temperature_max': args.temperature_max,
        'seed': args.seed,
        'model_path': None
    }

    evaluation_params = {
        'device': args.device
    }

    filtering_params = {
        'min_openknot_score': args.min_openknot_score,
        'max_shape_sc_error': args.max_shape_sc_error,
        'max_perplexity': args.max_perplexity,
        'min_gc_content': args.min_gc_content,
        'max_gc_content': args.max_gc_content,
        'top_n_per_target': args.top_n_per_target
    }

    # Run pipeline
    try:
        results = run_batch_design_pipeline(
            targets=targets,
            design_params=design_params,
            evaluation_params=evaluation_params,
            filtering_params=filtering_params,
            output_dir=args.output_dir,
            max_workers=args.max_workers
        )

        print(f"\n🎊 Successfully completed batch design pipeline!")
        print(f"Generated designs for {results['summary']['successful_targets']} targets")
        print(f"Total designs: {results['summary']['total_designs']}")
        print(f"Designs after filtering: {results['summary']['total_filtered']}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())