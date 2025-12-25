#!/usr/bin/env python3
"""
Script: batch_rna_pipeline.py
Description: High-throughput RNA design pipeline for multiple targets with evaluation and filtering

Original Use Case: examples/use_case_4_batch_design_pipeline_fixed.py
Dependencies Removed: Simplified pipeline, uses other clean scripts as modules

Usage:
    python scripts/batch_rna_pipeline.py --targets_file targets.csv --output_dir batch_results/ --n_designs_per_target 100

Example:
    python scripts/batch_rna_pipeline.py --targets_file examples/data/sequences/sample_targets.csv --output_dir results/batch --max_workers 2
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import json
import glob
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Essential scientific packages
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "n_designs_per_target": 100,
    "total_samples": 1000,
    "batch_size": 32,
    "max_workers": None,  # None = auto-detect
    "evaluation_enabled": True,
    "filtering_enabled": True,
    "min_score_threshold": 0.1,
    "max_results_per_target": 10,
    "verbose": True,
    "save_intermediate": True
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def validate_target(target: Dict[str, Any]) -> bool:
    """Validate a design target configuration."""
    required_fields = ['name']

    # Check required fields
    for field in required_fields:
        if field not in target:
            print(f"❌ Target missing required field: {field}")
            return False

    # Validate mode-specific requirements
    mode = target.get('mode', '2d')
    if mode == '3d':
        if not target.get('pdb_file'):
            print(f"❌ Target {target['name']}: 3D mode requires pdb_file")
            return False
        pdb_path = Path(target['pdb_file'])
        if not pdb_path.exists():
            print(f"❌ Target {target['name']}: PDB file not found: {pdb_path}")
            return False
    elif mode == '2d':
        if not target.get('secondary_structure'):
            print(f"❌ Target {target['name']}: 2D mode requires secondary_structure")
            return False

    return True


def load_targets_from_csv(csv_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load design targets from CSV file."""
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"Targets file not found: {csv_file}")

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

        # Auto-correct mode if needed
        if target['mode'] == '3d' and not target['pdb_file']:
            print(f"⚠️  Target {target['name']}: Changed to 2D mode (no PDB file)")
            target['mode'] = '2d'

        # Validate target
        if validate_target(target):
            targets.append(target)
        else:
            print(f"⚠️  Skipping invalid target: {target['name']}")

    return targets


def load_targets_from_directory(pdb_dir: Union[str, Path], mode: str = '3d') -> List[Dict[str, Any]]:
    """Load design targets from directory of PDB files."""
    pdb_dir = Path(pdb_dir)
    if not pdb_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")

    pdb_files = list(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise ValueError(f"No PDB files found in: {pdb_dir}")

    targets = []
    for pdb_file in pdb_files:
        target = {
            'name': pdb_file.stem,
            'pdb_file': str(pdb_file),
            'mode': mode,
            'description': f"Target from {pdb_file.name}"
        }

        if validate_target(target):
            targets.append(target)

    return targets


def import_clean_scripts():
    """Import functions from clean scripts."""
    import sys
    from pathlib import Path

    # Add scripts directory to path
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))

    try:
        # Import clean script functions
        from rna_inverse_design import run_rna_inverse_design
        from rna_evaluation import run_rna_evaluation
        from rna_structure_analysis import run_rna_structure_analysis

        return {
            'design': run_rna_inverse_design,
            'evaluate': run_rna_evaluation,
            'analyze': run_rna_structure_analysis
        }
    except ImportError as e:
        raise ImportError(f"Could not import clean scripts: {e}. "
                         f"Make sure scripts are in: {scripts_dir}")


def design_single_target(target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Design RNA sequences for a single target."""
    try:
        functions = import_clean_scripts()

        # Run design
        design_config = {
            'total_samples': config.get('total_samples', 1000),
            'n_pass': config.get('n_designs_per_target', 100),
            'seed': 42,
            'verbose': False  # Quiet for batch processing
        }

        if target['mode'] == '3d':
            result = functions['design'](
                pdb_file=target['pdb_file'],
                mode='3d',
                config=design_config
            )
        else:
            result = functions['design'](
                secondary_structure=target['secondary_structure'],
                mode='2d',
                config=design_config
            )

        if result.get('success', True):
            return {
                'target_name': target['name'],
                'success': True,
                'sequences': result.get('sequences', []),
                'num_generated': len(result.get('sequences', [])),
                'metadata': result.get('metadata', {}),
                'error': None
            }
        else:
            return {
                'target_name': target['name'],
                'success': False,
                'sequences': [],
                'num_generated': 0,
                'error': result.get('error', 'Unknown error')
            }

    except Exception as e:
        return {
            'target_name': target['name'],
            'success': False,
            'sequences': [],
            'num_generated': 0,
            'error': str(e)
        }


def evaluate_designs(sequences: List[str], target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate generated designs."""
    try:
        if not sequences:
            return {'scores': [], 'summary': {'mean_score': 0.0}}

        functions = import_clean_scripts()

        # Use secondary structure for evaluation
        target_structure = target.get('secondary_structure', '.' * len(sequences[0]))

        eval_config = {
            'verbose': False
        }

        result = functions['evaluate'](
            sequences=sequences,
            target_structure=target_structure,
            config=eval_config
        )

        return {
            'evaluation_df': result.get('results_df'),
            'summary': result.get('summary_stats', {}),
            'success': True
        }

    except Exception as e:
        return {
            'evaluation_df': None,
            'summary': {'mean_score': 0.0},
            'success': False,
            'error': str(e)
        }


def filter_and_rank_designs(evaluation_df, config: Dict[str, Any]) -> Dict[str, Any]:
    """Filter and rank designs based on scores."""
    if evaluation_df is None or evaluation_df.empty:
        return {'filtered_df': None, 'top_designs': []}

    try:
        # Simple filtering based on basic metrics
        filtered_df = evaluation_df.copy()

        # Filter by length (remove very short or very long sequences)
        if 'length' in filtered_df.columns:
            length_mean = filtered_df['length'].mean()
            length_std = filtered_df['length'].std()
            filtered_df = filtered_df[
                (filtered_df['length'] >= length_mean - 2*length_std) &
                (filtered_df['length'] <= length_mean + 2*length_std)
            ]

        # Filter by GC content (reasonable range)
        if 'gc_content' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['gc_content'] >= 0.2) &
                (filtered_df['gc_content'] <= 0.8)
            ]

        # Rank by multiple criteria (simple scoring)
        score_columns = ['openknot_score', 'sc_score_ribonanzanet', 'sc_score_ribonanzanet_ss']
        available_scores = [col for col in score_columns if col in filtered_df.columns]

        if available_scores:
            # Calculate combined score
            filtered_df['combined_score'] = filtered_df[available_scores].mean(axis=1)
            # Sort by combined score (descending)
            filtered_df = filtered_df.sort_values('combined_score', ascending=False)
        else:
            # Fall back to GC content ranking
            if 'gc_content' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('gc_content', ascending=False)

        # Get top designs
        max_results = config.get('max_results_per_target', 10)
        top_designs = filtered_df.head(max_results).to_dict('records')

        return {
            'filtered_df': filtered_df,
            'top_designs': top_designs,
            'num_passed_filters': len(filtered_df),
            'filtering_success': True
        }

    except Exception as e:
        return {
            'filtered_df': evaluation_df,
            'top_designs': evaluation_df.head(config.get('max_results_per_target', 10)).to_dict('records'),
            'num_passed_filters': len(evaluation_df),
            'filtering_success': False,
            'error': str(e)
        }


def process_single_target(target: Dict[str, Any], config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Complete pipeline for a single target."""
    target_name = target['name']

    if config.get('verbose', True):
        print(f"🎯 Processing target: {target_name}")

    # Create target-specific output directory
    target_output_dir = output_dir / f"target_{target_name}"
    target_output_dir.mkdir(exist_ok=True)

    results = {
        'target': target,
        'design_results': {},
        'evaluation_results': {},
        'filtering_results': {},
        'summary': {},
        'output_files': {}
    }

    # Phase 1: Design Generation
    design_result = design_single_target(target, config)
    results['design_results'] = design_result

    if not design_result['success']:
        if config.get('verbose', True):
            print(f"❌ Design failed for {target_name}: {design_result['error']}")
        return results

    sequences = design_result['sequences']
    if config.get('verbose', True):
        print(f"✅ Generated {len(sequences)} sequences for {target_name}")

    # Save design results
    design_df = pd.DataFrame({'sequence': sequences})
    design_file = target_output_dir / f"designs_{target_name}.csv"
    design_df.to_csv(design_file, index=False)
    results['output_files']['designs'] = str(design_file)

    # Phase 2: Evaluation (if enabled)
    if config.get('evaluation_enabled', True):
        eval_result = evaluate_designs(sequences, target, config)
        results['evaluation_results'] = eval_result

        if eval_result['success'] and eval_result['evaluation_df'] is not None:
            eval_file = target_output_dir / f"evaluation_{target_name}.csv"
            eval_result['evaluation_df'].to_csv(eval_file, index=False)
            results['output_files']['evaluation'] = str(eval_file)

            # Phase 3: Filtering (if enabled)
            if config.get('filtering_enabled', True):
                filter_result = filter_and_rank_designs(eval_result['evaluation_df'], config)
                results['filtering_results'] = filter_result

                if filter_result['filtering_success']:
                    # Save filtered results
                    filtered_file = target_output_dir / f"filtered_{target_name}.csv"
                    filter_result['filtered_df'].to_csv(filtered_file, index=False)
                    results['output_files']['filtered'] = str(filtered_file)

                    # Save top designs
                    top_file = target_output_dir / f"top_{target_name}.json"
                    with open(top_file, 'w') as f:
                        json.dump(filter_result['top_designs'], f, indent=2, default=str)
                    results['output_files']['top_designs'] = str(top_file)

    # Create summary
    results['summary'] = {
        'target_name': target_name,
        'design_success': design_result['success'],
        'sequences_generated': len(sequences),
        'evaluation_completed': results['evaluation_results'].get('success', False),
        'filtering_completed': results['filtering_results'].get('filtering_success', False),
        'top_designs_count': len(results['filtering_results'].get('top_designs', [])),
        'output_directory': str(target_output_dir)
    }

    if config.get('verbose', True):
        print(f"✅ Completed {target_name}: {results['summary']['sequences_generated']} sequences, "
              f"{results['summary']['top_designs_count']} top designs")

    return results


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_rna_pipeline(
    targets: Union[List[Dict], str, Path],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run high-throughput RNA design pipeline for multiple targets.

    Args:
        targets: List of target dicts, or path to CSV file, or directory with PDB files
        output_dir: Directory to save all results
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: Per-target results
            - summary: Overall pipeline summary
            - config_used: Configuration used
            - output_dir: Output directory path
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_rna_pipeline(
        ...     targets="targets.csv",
        ...     output_dir="results/batch",
        ...     n_designs_per_target=100
        ... )
        >>> print(f"Processed {len(result['results'])} targets")
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load targets
    if isinstance(targets, (str, Path)):
        targets_path = Path(targets)
        if targets_path.is_file() and targets_path.suffix == '.csv':
            targets_list = load_targets_from_csv(targets_path)
        elif targets_path.is_dir():
            targets_list = load_targets_from_directory(targets_path)
        else:
            raise ValueError(f"Invalid targets path: {targets_path}")
    else:
        targets_list = targets

    if config.get('verbose', True):
        print(f"🚀 Starting batch pipeline for {len(targets_list)} targets")
        print(f"📁 Output directory: {output_dir}")

    # Process targets
    max_workers = config.get('max_workers') or min(len(targets_list), mp.cpu_count())

    if max_workers == 1 or len(targets_list) == 1:
        # Serial processing
        results = []
        for i, target in enumerate(targets_list):
            if config.get('verbose', True):
                print(f"\n--- Target {i+1}/{len(targets_list)} ---")
            result = process_single_target(target, config, output_dir)
            results.append(result)
    else:
        # Parallel processing
        if config.get('verbose', True):
            print(f"🔄 Using {max_workers} workers for parallel processing")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_target = {
                executor.submit(process_single_target, target, config, output_dir): target
                for target in targets_list
            }

            # Collect results
            for future in as_completed(future_to_target):
                target = future_to_target[future]
                try:
                    result = future.result()
                    results.append(result)
                    if config.get('verbose', True):
                        print(f"✅ Completed {target['name']}")
                except Exception as e:
                    print(f"❌ Failed {target['name']}: {e}")
                    # Add error result
                    results.append({
                        'target': target,
                        'summary': {
                            'target_name': target['name'],
                            'design_success': False,
                            'error': str(e)
                        }
                    })

    # Create overall summary
    summary = {
        'total_targets': len(targets_list),
        'successful_designs': sum(1 for r in results if r['summary'].get('design_success', False)),
        'total_sequences': sum(r['summary'].get('sequences_generated', 0) for r in results),
        'total_top_designs': sum(r['summary'].get('top_designs_count', 0) for r in results),
        'pipeline_config': config,
        'timestamp': datetime.now().isoformat()
    }

    # Save pipeline results
    pipeline_file = output_dir / "pipeline_results.json"
    with open(pipeline_file, 'w') as f:
        json.dump({
            'summary': summary,
            'results': [r['summary'] for r in results],
            'config': config
        }, f, indent=2, default=str)

    # Save detailed results summary
    summary_df = pd.DataFrame([r['summary'] for r in results])
    summary_file = output_dir / "pipeline_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    if config.get('verbose', True):
        print(f"\n🎉 Pipeline complete!")
        print(f"📊 Summary: {summary['successful_designs']}/{summary['total_targets']} targets successful")
        print(f"🧬 Total sequences: {summary['total_sequences']}")
        print(f"⭐ Total top designs: {summary['total_top_designs']}")
        print(f"📁 Results saved to: {output_dir}")

    return {
        'results': results,
        'summary': summary,
        'config_used': config,
        'output_dir': str(output_dir),
        'metadata': {
            'pipeline_complete': True,
            'timestamp': datetime.now().isoformat(),
            'pipeline_file': str(pipeline_file),
            'summary_file': str(summary_file)
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
    parser.add_argument('--targets_file', help='CSV file with target specifications')
    parser.add_argument('--pdb_dir', help='Directory containing PDB files')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--config', help='Config file (JSON)')

    # Pipeline parameters
    parser.add_argument('--n_designs_per_target', type=int, help='Number of designs per target')
    parser.add_argument('--max_workers', type=int, help='Maximum parallel workers')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing')

    # Processing options
    parser.add_argument('--no_evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--no_filtering', action='store_true', help='Skip filtering phase')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Determine targets input
    if args.targets_file:
        targets_input = args.targets_file
    elif args.pdb_dir:
        targets_input = args.pdb_dir
    else:
        parser.error("Must provide either --targets_file or --pdb_dir")

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Add CLI options to config
    cli_options = {
        'n_designs_per_target': args.n_designs_per_target,
        'max_workers': args.max_workers,
        'batch_size': args.batch_size,
        'evaluation_enabled': not args.no_evaluation,
        'filtering_enabled': not args.no_filtering,
        'verbose': args.verbose
    }

    for key, value in cli_options.items():
        if value is not None:
            config[key] = value

    # Run pipeline
    try:
        result = run_batch_rna_pipeline(
            targets=targets_input,
            output_dir=args.output_dir,
            config=config
        )

        print(f"\n✅ Batch pipeline completed successfully!")
        print(f"📁 Results: {result['output_dir']}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())