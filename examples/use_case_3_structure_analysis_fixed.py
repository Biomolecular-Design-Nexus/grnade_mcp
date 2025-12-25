#!/usr/bin/env python3
"""
Use Case 3: RNA Secondary Structure Analysis and Visualization

This script provides comprehensive RNA secondary structure analysis including:
- Secondary structure prediction using EternaFold
- Dot-bracket notation parsing and validation
- Pseudoknot structure handling
- Structure visualization
- PDB secondary structure extraction

Usage:
    python examples/use_case_3_structure_analysis.py --sequence "GGGGAAAACCCC" --output analysis_results.json
    python examples/use_case_3_structure_analysis.py --pdb examples/data/RNASolo.pdb --visualize

Dependencies:
    - EternaFold (conda install eternafold)
    - Biotite
    - All gRNAde dependencies

Author: gRNAde MCP
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union

# gRNAde imports
import sys
sys.path.append('repo/geometric-rna-design')

try:
    from src.data.sec_struct_utils import (
        predict_sec_struct,
        pdb_to_sec_struct,
        dotbracket_to_adjacency,
        dotbracket_to_paired,
        validate_dotbracket,
        get_paired_positions,
        get_pseudoknot_order
    )
    from src.data.viz_utils import draw_2d_struct, print_rna_data
    from src.data.data_utils import pdb_to_tensor
except ImportError as e:
    print(f"Error importing gRNAde modules: {e}")
    print("Please make sure you're running this from the MCP root directory")
    exit(1)


def analyze_rna_structure(
    sequence: Optional[str] = None,
    pdb_file: Optional[str] = None,
    secondary_structure: Optional[str] = None,
    predict_structure: bool = True,
    include_pseudoknots: bool = True,
    visualize: bool = False,
    output_file: Optional[str] = None
) -> Dict:
    """
    Comprehensive RNA secondary structure analysis.

    Args:
        sequence: RNA sequence to analyze
        pdb_file: Path to PDB file for structure extraction
        secondary_structure: Known secondary structure in dot-bracket notation
        predict_structure: Whether to predict secondary structure using EternaFold
        include_pseudoknots: Whether to include pseudoknot analysis
        visualize: Whether to generate structure visualizations
        output_file: Optional path to save results

    Returns:
        dict: Comprehensive analysis results
    """

    print("🔬 RNA Secondary Structure Analysis")

    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'secondary_structure',
        'input_data': {},
        'structure_analysis': {},
        'predictions': {},
        'visualizations': [],
        'summary': {}
    }

    # Load input data
    if pdb_file:
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        print(f"📁 Loading structure from PDB: {pdb_file}")

        # Extract sequence and coordinates
        try:
            coords, seq, sec_struct, sasa = pdb_to_tensor(pdb_file)
            if sequence is None:
                sequence = seq
            if secondary_structure is None:
                secondary_structure = sec_struct

            results['input_data']['pdb_file'] = pdb_file
            results['input_data']['extracted_sequence'] = seq
            results['input_data']['extracted_structure'] = sec_struct
            results['input_data']['coordinates_shape'] = coords.shape if coords is not None else None
            results['input_data']['sasa_available'] = sasa is not None

            print(f"✅ Extracted sequence length: {len(seq)}")
            print(f"✅ Secondary structure: {sec_struct[:50]}...")

        except Exception as e:
            print(f"❌ Error extracting from PDB: {e}")
            # Try alternative extraction method
            try:
                secondary_structure = pdb_to_sec_struct(pdb_file)
                results['input_data']['pdb_extraction_method'] = 'sec_struct_utils'
                results['input_data']['extracted_structure'] = secondary_structure
            except Exception as e2:
                print(f"❌ Alternative extraction also failed: {e2}")
                return results

    # Validate sequence
    if not sequence:
        raise ValueError("No sequence provided")

    # Clean and validate sequence
    sequence = sequence.upper().replace('T', 'U')
    valid_nucs = set('AUGC')
    if not all(nuc in valid_nucs for nuc in sequence):
        invalid_chars = set(sequence) - valid_nucs
        print(f"⚠️  Warning: Invalid nucleotides found: {invalid_chars}")
        # Filter to valid nucleotides only
        sequence = ''.join([nuc for nuc in sequence if nuc in valid_nucs])

    results['input_data']['sequence'] = sequence
    results['input_data']['length'] = len(sequence)
    results['input_data']['gc_content'] = (sequence.count('G') + sequence.count('C')) / len(sequence)

    print(f"📝 Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
    print(f"📊 Length: {len(sequence)} nucleotides")
    print(f"📊 GC content: {results['input_data']['gc_content']:.2%}")

    # Structure prediction
    if predict_structure and secondary_structure is None:
        print("🔮 Predicting secondary structure with EternaFold...")
        try:
            predicted_structure = predict_sec_struct(sequence, method='eternafold')
            results['predictions']['eternafold_structure'] = predicted_structure
            results['predictions']['eternafold_method'] = 'eternafold'

            if secondary_structure is None:
                secondary_structure = predicted_structure

            print(f"✅ Predicted structure: {predicted_structure}")

        except Exception as e:
            print(f"❌ Structure prediction failed: {e}")
            print("⚠️  Continuing without structure prediction")
            results['predictions']['error'] = str(e)

    # Analyze secondary structure
    if secondary_structure:
        print("📐 Analyzing secondary structure...")

        # Validate dot-bracket notation
        try:
            is_valid, error_msg = validate_dotbracket(secondary_structure)
            results['structure_analysis']['valid_dotbracket'] = is_valid
            if not is_valid:
                results['structure_analysis']['validation_error'] = error_msg
                print(f"❌ Invalid dot-bracket notation: {error_msg}")
        except:
            results['structure_analysis']['valid_dotbracket'] = False
            print("❌ Could not validate dot-bracket notation")

        # Basic structure statistics
        paired_positions = get_paired_positions(secondary_structure)
        results['structure_analysis']['paired_positions'] = len(paired_positions)
        results['structure_analysis']['unpaired_positions'] = len(secondary_structure) - len(paired_positions) * 2
        results['structure_analysis']['pairing_percentage'] = (len(paired_positions) * 2) / len(secondary_structure)

        # Convert to adjacency matrix
        try:
            adj_matrix = dotbracket_to_adjacency(secondary_structure)
            results['structure_analysis']['adjacency_matrix_shape'] = adj_matrix.shape
            results['structure_analysis']['total_base_pairs'] = int(np.sum(adj_matrix) // 2)
        except Exception as e:
            print(f"❌ Error creating adjacency matrix: {e}")
            results['structure_analysis']['adjacency_error'] = str(e)

        # Analyze paired/unpaired regions
        try:
            paired_array = dotbracket_to_paired(secondary_structure)
            results['structure_analysis']['paired_array_length'] = len(paired_array)
            results['structure_analysis']['stems'] = []
            results['structure_analysis']['loops'] = []

            # Find stems and loops
            in_stem = False
            current_stem = []
            current_loop = []

            for i, is_paired in enumerate(paired_array):
                if is_paired and not in_stem:
                    if current_loop:
                        results['structure_analysis']['loops'].append(len(current_loop))
                        current_loop = []
                    in_stem = True
                    current_stem = [i]
                elif is_paired and in_stem:
                    current_stem.append(i)
                elif not is_paired and in_stem:
                    if current_stem:
                        results['structure_analysis']['stems'].append(len(current_stem))
                        current_stem = []
                    in_stem = False
                    current_loop = [i]
                elif not is_paired and not in_stem:
                    current_loop.append(i)

            # Close final region
            if current_stem:
                results['structure_analysis']['stems'].append(len(current_stem))
            if current_loop:
                results['structure_analysis']['loops'].append(len(current_loop))

            results['structure_analysis']['num_stems'] = len(results['structure_analysis']['stems'])
            results['structure_analysis']['num_loops'] = len(results['structure_analysis']['loops'])
            results['structure_analysis']['avg_stem_length'] = np.mean(results['structure_analysis']['stems']) if results['structure_analysis']['stems'] else 0
            results['structure_analysis']['avg_loop_length'] = np.mean(results['structure_analysis']['loops']) if results['structure_analysis']['loops'] else 0

        except Exception as e:
            print(f"❌ Error analyzing paired regions: {e}")
            results['structure_analysis']['paired_analysis_error'] = str(e)

        # Pseudoknot analysis
        if include_pseudoknots:
            print("🔗 Analyzing pseudoknot structures...")
            try:
                pk_order = get_pseudoknot_order(secondary_structure)
                results['structure_analysis']['pseudoknot_order'] = pk_order
                results['structure_analysis']['has_pseudoknots'] = pk_order > 0
                results['structure_analysis']['pseudoknot_complexity'] = pk_order

                if pk_order > 0:
                    print(f"✅ Pseudoknots detected (order {pk_order})")
                else:
                    print("📝 No pseudoknots found")

            except Exception as e:
                print(f"❌ Error analyzing pseudoknots: {e}")
                results['structure_analysis']['pseudoknot_error'] = str(e)

        # Visualization
        if visualize:
            print("🎨 Generating structure visualizations...")
            try:
                # Generate 2D structure diagram
                viz_path = f"structure_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                draw_2d_struct(sequence, secondary_structure, save_path=viz_path)

                results['visualizations'].append({
                    'type': '2d_structure',
                    'file_path': viz_path,
                    'description': 'Secondary structure diagram'
                })

                print(f"📊 Saved visualization: {viz_path}")

            except Exception as e:
                print(f"❌ Visualization failed: {e}")
                results['visualizations'].append({
                    'type': '2d_structure',
                    'error': str(e)
                })

    # Generate summary
    results['summary'] = {
        'sequence_length': len(sequence),
        'gc_content': results['input_data']['gc_content'],
        'has_structure': secondary_structure is not None,
        'structure_predicted': 'eternafold_structure' in results['predictions'],
        'has_pseudoknots': results['structure_analysis'].get('has_pseudoknots', False),
        'pseudoknot_order': results['structure_analysis'].get('pseudoknot_order', 0),
        'pairing_percentage': results['structure_analysis'].get('pairing_percentage', 0),
        'num_base_pairs': results['structure_analysis'].get('total_base_pairs', 0),
        'num_stems': results['structure_analysis'].get('num_stems', 0),
        'num_loops': results['structure_analysis'].get('num_loops', 0),
        'valid_structure': results['structure_analysis'].get('valid_dotbracket', False)
    }

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📁 Results saved to: {output_file}")

    print("\n📈 Analysis Summary:")
    print("=" * 50)
    print(f"Sequence length: {results['summary']['sequence_length']} nucleotides")
    print(f"GC content: {results['summary']['gc_content']:.2%}")
    print(f"Base pairs: {results['summary']['num_base_pairs']}")
    print(f"Pairing percentage: {results['summary']['pairing_percentage']:.2%}")
    print(f"Stems: {results['summary']['num_stems']}")
    print(f"Loops: {results['summary']['num_loops']}")
    print(f"Has pseudoknots: {results['summary']['has_pseudoknots']}")
    if results['summary']['has_pseudoknots']:
        print(f"Pseudoknot order: {results['summary']['pseudoknot_order']}")

    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='RNA Secondary Structure Analysis and Visualization')

    # Input options
    parser.add_argument('--sequence', '-s', type=str,
                       help='RNA sequence to analyze')
    parser.add_argument('--pdb', type=str,
                       help='PDB file to extract sequence and structure from')
    parser.add_argument('--structure', '--ss', type=str,
                       help='Known secondary structure in dot-bracket notation')

    # Analysis options
    parser.add_argument('--predict', action='store_true', default=True,
                       help='Predict secondary structure using EternaFold (default: True)')
    parser.add_argument('--no-predict', dest='predict', action='store_false',
                       help='Skip structure prediction')
    parser.add_argument('--pseudoknots', action='store_true', default=True,
                       help='Include pseudoknot analysis (default: True)')
    parser.add_argument('--no-pseudoknots', dest='pseudoknots', action='store_false',
                       help='Skip pseudoknot analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate structure visualizations')

    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON format)')

    args = parser.parse_args()

    # Validate inputs
    if not args.sequence and not args.pdb:
        parser.error("Either --sequence or --pdb must be provided")

    # Run analysis
    try:
        results = analyze_rna_structure(
            sequence=args.sequence,
            pdb_file=args.pdb,
            secondary_structure=args.structure,
            predict_structure=args.predict,
            include_pseudoknots=args.pseudoknots,
            visualize=args.visualize,
            output_file=args.output
        )

        print("\n🎉 RNA structure analysis completed successfully!")

        if results['summary']['has_structure']:
            print(f"Structure contains {results['summary']['num_base_pairs']} base pairs")
            print(f"Pairing efficiency: {results['summary']['pairing_percentage']:.1%}")

            if results['summary']['has_pseudoknots']:
                print(f"⚠️  Complex structure with pseudoknots (order {results['summary']['pseudoknot_order']})")
            else:
                print("✅ Simple nested structure (no pseudoknots)")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())