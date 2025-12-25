#!/usr/bin/env python3
"""
Use Case 3: RNA Secondary Structure Analysis - Minimal Version

This script provides basic RNA secondary structure analysis including:
- Secondary structure prediction using EternaFold
- Dot-bracket notation parsing and validation
- Basic structure statistics

Usage:
    python examples/use_case_3_structure_analysis_minimal.py --sequence "GGGGAAAACCCC" --predict --output analysis_results.json

Dependencies:
    - All gRNAde dependencies
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
        dotbracket_to_paired
    )
except ImportError as e:
    print(f"Error importing gRNAde modules: {e}")
    print("Please make sure you're running this from the MCP root directory")
    exit(1)


def validate_dotbracket(structure: str) -> Tuple[bool, str]:
    """
    Simple dot-bracket validation function.
    """
    try:
        stack = []
        for char in structure:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False, "Unmatched closing parenthesis"
                stack.pop()
            elif char not in '.()[]{}':
                return False, f"Invalid character: {char}"

        if stack:
            return False, "Unmatched opening parenthesis"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


def get_paired_positions(structure: str) -> List[Tuple[int, int]]:
    """
    Get paired positions from dot-bracket notation.
    """
    paired_positions = []
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                paired_positions.append((j, i))

    return paired_positions


def get_pseudoknot_order(structure: str) -> int:
    """
    Determine pseudoknot order (simplified version).
    """
    # This is a simplified implementation
    # Real pseudoknot analysis would be more complex
    if '[' in structure and ']' in structure:
        return 1
    return 0


def analyze_structure_properties(structure: str) -> Dict:
    """
    Analyze basic properties of a secondary structure.
    """
    paired_positions = get_paired_positions(structure)
    unpaired_count = structure.count('.')
    paired_count = len(structure) - unpaired_count

    return {
        "length": len(structure),
        "paired_positions": len(paired_positions),
        "unpaired_positions": unpaired_count,
        "pairing_percentage": paired_count / len(structure) if len(structure) > 0 else 0,
        "total_base_pairs": len(paired_positions),
        "has_pseudoknots": '[' in structure or ']' in structure,
        "pseudoknot_order": get_pseudoknot_order(structure)
    }


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
    Main analysis function for RNA structure.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "input": {},
        "structure_analysis": {},
        "predictions": {},
        "summary": {}
    }

    # Record input parameters
    results["input"] = {
        "sequence": sequence,
        "pdb_file": pdb_file,
        "secondary_structure": secondary_structure,
        "predict_structure": predict_structure,
        "include_pseudoknots": include_pseudoknots
    }

    try:
        # Get sequence and structure
        if sequence is None and pdb_file is None and secondary_structure is None:
            raise ValueError("Must provide sequence, PDB file, or secondary structure")

        # Handle different input types
        if pdb_file and sequence:
            print(f"📄 Analyzing PDB structure: {pdb_file}")
            try:
                predicted_structure = pdb_to_sec_struct(pdb_file, sequence)
                results["predictions"]["from_pdb"] = predicted_structure
                secondary_structure = predicted_structure
            except Exception as e:
                print(f"⚠️ Warning: Could not extract structure from PDB: {e}")
                print("Proceeding with sequence-only analysis")

        # Predict structure if requested
        if predict_structure and sequence:
            print(f"🔬 Predicting secondary structure for sequence: {sequence[:50]}...")
            try:
                predicted_structure = predict_sec_struct(sequence)
                results["predictions"]["eternafold"] = predicted_structure
                if secondary_structure is None:
                    secondary_structure = predicted_structure
            except Exception as e:
                print(f"⚠️ Warning: Structure prediction failed: {e}")

        # Analyze structure if we have one
        if secondary_structure:
            print(f"📊 Analyzing structure: {secondary_structure[:50]}...")

            # Validate structure
            is_valid, error_msg = validate_dotbracket(secondary_structure)

            if is_valid:
                structure_props = analyze_structure_properties(secondary_structure)
                results["structure_analysis"] = {
                    "valid_dotbracket": True,
                    **structure_props
                }
                print(f"✅ Structure analysis complete")
                print(f"   Length: {structure_props['length']}")
                print(f"   Base pairs: {structure_props['total_base_pairs']}")
                print(f"   Pairing: {structure_props['pairing_percentage']:.1%}")
            else:
                results["structure_analysis"] = {
                    "valid_dotbracket": False,
                    "error": error_msg
                }
                print(f"❌ Invalid structure: {error_msg}")

        # Summary
        results["summary"] = {
            "analysis_successful": bool(results["structure_analysis"]),
            "has_structure": secondary_structure is not None,
            "has_prediction": bool(results["predictions"]),
            "total_runtime_info": "Analysis completed"
        }

        # Save results if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {output_file}")

        return results

    except Exception as e:
        error_results = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "input": results["input"]
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(error_results, f, indent=2)

        print(f"❌ Analysis failed: {e}")
        return error_results


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="RNA Secondary Structure Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python use_case_3_structure_analysis_minimal.py --sequence "GGGGAAAACCCC" --predict
  python use_case_3_structure_analysis_minimal.py --pdb examples/data/structures/RNASolo.pdb --sequence "GGGGAAAACCCC"
  python use_case_3_structure_analysis_minimal.py --secondary_structure "((((....))))"
        """
    )

    # Input options
    parser.add_argument('--sequence', type=str, help='RNA sequence to analyze')
    parser.add_argument('--pdb', type=str, help='PDB file containing RNA structure')
    parser.add_argument('--secondary_structure', type=str, help='Secondary structure in dot-bracket notation')

    # Analysis options
    parser.add_argument('--predict', action='store_true', help='Predict secondary structure using EternaFold')
    parser.add_argument('--include_pseudoknots', action='store_true', default=True, help='Include pseudoknot analysis')
    parser.add_argument('--visualize', action='store_true', help='Generate structure visualization (if available)')

    # Output options
    parser.add_argument('--output', type=str, help='Output file for analysis results (JSON format)')

    args = parser.parse_args()

    # Validate inputs
    if not any([args.sequence, args.pdb, args.secondary_structure]):
        parser.error("Must provide at least one of: --sequence, --pdb, or --secondary_structure")

    print("🧬 Starting RNA Structure Analysis")
    print("=" * 50)

    # Run analysis
    results = analyze_rna_structure(
        sequence=args.sequence,
        pdb_file=args.pdb,
        secondary_structure=args.secondary_structure,
        predict_structure=args.predict,
        include_pseudoknots=args.include_pseudoknots,
        visualize=args.visualize,
        output_file=args.output
    )

    # Print summary
    print("\n📋 Analysis Summary:")
    print("=" * 50)

    if results.get("structure_analysis"):
        analysis = results["structure_analysis"]
        if analysis.get("valid_dotbracket"):
            print(f"✅ Valid structure analyzed")
            print(f"   Length: {analysis.get('length', 'N/A')}")
            print(f"   Base pairs: {analysis.get('total_base_pairs', 'N/A')}")
            print(f"   Pairing: {analysis.get('pairing_percentage', 0):.1%}")
            print(f"   Has pseudoknots: {analysis.get('has_pseudoknots', False)}")
        else:
            print(f"❌ Structure validation failed: {analysis.get('error', 'Unknown error')}")

    if results.get("predictions"):
        print(f"🔬 Predictions generated: {list(results['predictions'].keys())}")

    if results.get("error"):
        print(f"❌ Error occurred: {results['error']}")
        return 1

    print("\n🎉 Analysis complete!")
    return 0


if __name__ == '__main__':
    exit(main())