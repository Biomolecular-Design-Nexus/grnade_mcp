#!/usr/bin/env python3
"""
Script: rna_structure_analysis.py
Description: Analyze RNA secondary structures - validation, statistics, and property calculation

Original Use Case: examples/use_case_3_structure_analysis_minimal.py
Dependencies Removed: Fully self-contained, no repo dependencies

Usage:
    python scripts/rna_structure_analysis.py --secondary_structure "((((....))))" --output analysis_results.json

Example:
    python scripts/rna_structure_analysis.py --secondary_structure "((((....))))" --output results/analysis.json
    python scripts/rna_structure_analysis.py --sequence "GGGAAACCC" --predict --output results/predicted.json
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
from datetime import datetime

# Essential scientific packages
import numpy as np

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "include_pseudoknots": True,
    "validate_structure": True,
    "include_statistics": True,
    "verbose": True
}

# ==============================================================================
# Inlined Structure Analysis Functions (fully self-contained)
# ==============================================================================
def validate_dotbracket(structure: str) -> Tuple[bool, str]:
    """
    Validate dot-bracket notation structure.

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        stack = []
        bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
        opening = set(bracket_pairs.keys())
        closing = set(bracket_pairs.values())

        for i, char in enumerate(structure):
            if char in opening:
                stack.append((char, i))
            elif char in closing:
                if not stack:
                    return False, f"Unmatched closing bracket '{char}' at position {i}"

                open_char, open_pos = stack.pop()
                expected_close = bracket_pairs[open_char]
                if char != expected_close:
                    return False, f"Mismatched brackets: '{open_char}' at {open_pos} and '{char}' at {i}"
            elif char not in '.':
                return False, f"Invalid character '{char}' at position {i}"

        if stack:
            open_char, open_pos = stack[0]
            return False, f"Unmatched opening bracket '{open_char}' at position {open_pos}"

        return True, "Valid dot-bracket notation"

    except Exception as e:
        return False, f"Validation error: {e}"


def get_paired_positions(structure: str) -> List[Tuple[int, int]]:
    """
    Get all paired positions from dot-bracket notation.

    Returns:
        List of (i, j) tuples representing base pairs
    """
    paired_positions = []
    stacks = {'(': [], '[': [], '{': [], '<': []}
    bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}

    for i, char in enumerate(structure):
        if char in stacks:
            stacks[char].append(i)
        elif char in bracket_pairs.values():
            # Find corresponding opening bracket
            for open_char, close_char in bracket_pairs.items():
                if char == close_char and stacks[open_char]:
                    j = stacks[open_char].pop()
                    paired_positions.append((j, i))
                    break

    return paired_positions


def get_pseudoknot_order(structure: str) -> int:
    """
    Determine the pseudoknot order of a structure.

    Returns:
        Integer representing the highest order pseudoknot (0 = no pseudoknots)
    """
    # Simple pseudoknot detection
    bracket_types = ['()', '[]', '{}', '<>']

    # Check for nested brackets of different types
    found_brackets = []
    for bracket_type in bracket_types:
        if bracket_type[0] in structure and bracket_type[1] in structure:
            found_brackets.append(bracket_type)

    # If more than one bracket type, potential pseudoknot
    if len(found_brackets) > 1:
        return len(found_brackets) - 1

    return 0


def analyze_structure_properties(structure: str) -> Dict[str, Any]:
    """
    Analyze comprehensive properties of a secondary structure.

    Returns:
        Dictionary with structure statistics and properties
    """
    if not structure:
        return {}

    # Basic counts
    length = len(structure)
    unpaired_count = structure.count('.')

    # Get paired positions
    paired_positions = get_paired_positions(structure)
    num_base_pairs = len(paired_positions)

    # Calculate pairing statistics
    paired_count = length - unpaired_count
    pairing_percentage = paired_count / length if length > 0 else 0.0

    # Pseudoknot analysis
    has_pseudoknots = any(char in structure for char in '[]{}><')
    pseudoknot_order = get_pseudoknot_order(structure)

    # Stem analysis (consecutive base pairs)
    stems = analyze_stems(paired_positions)

    # Loop analysis
    loops = analyze_loops(structure, paired_positions)

    return {
        "length": length,
        "paired_positions": num_base_pairs,
        "unpaired_positions": unpaired_count,
        "pairing_percentage": pairing_percentage,
        "total_base_pairs": num_base_pairs,
        "has_pseudoknots": has_pseudoknots,
        "pseudoknot_order": pseudoknot_order,
        "num_stems": len(stems),
        "avg_stem_length": np.mean([stem['length'] for stem in stems]) if stems else 0,
        "num_loops": len(loops),
        "loop_types": {loop_type: count for loop_type, count in loops.items()},
        "base_pair_list": paired_positions,
        "stems": stems
    }


def analyze_stems(paired_positions: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """
    Analyze stem structures from paired positions.

    Returns:
        List of stem dictionaries with start, end, and length
    """
    if not paired_positions:
        return []

    # Sort pairs by first position
    sorted_pairs = sorted(paired_positions)

    stems = []
    current_stem = []

    for i, (pos1, pos2) in enumerate(sorted_pairs):
        if i == 0:
            current_stem = [(pos1, pos2)]
        else:
            prev_pos1, prev_pos2 = sorted_pairs[i-1]

            # Check if this pair continues the current stem
            # (consecutive positions in both directions)
            if pos1 == prev_pos1 + 1 and pos2 == prev_pos2 - 1:
                current_stem.append((pos1, pos2))
            else:
                # End current stem, start new one
                if len(current_stem) >= 2:  # Minimum stem length
                    stems.append({
                        'start': current_stem[0],
                        'end': current_stem[-1],
                        'length': len(current_stem),
                        'pairs': current_stem
                    })
                current_stem = [(pos1, pos2)]

    # Add final stem
    if len(current_stem) >= 2:
        stems.append({
            'start': current_stem[0],
            'end': current_stem[-1],
            'length': len(current_stem),
            'pairs': current_stem
        })

    return stems


def analyze_loops(structure: str, paired_positions: List[Tuple[int, int]]) -> Dict[str, int]:
    """
    Analyze loop structures in the secondary structure.

    Returns:
        Dictionary with counts of different loop types
    """
    loops = {
        'hairpin': 0,
        'internal': 0,
        'bulge': 0,
        'multiloop': 0
    }

    # This is a simplified loop analysis
    # A full implementation would require more sophisticated parsing

    # Count apparent hairpin loops (simple heuristic)
    # Look for patterns like (((...)))
    for i in range(len(structure) - 6):  # Minimum hairpin size
        if structure[i:i+3] == '(((' and '))' in structure[i+3:i+10]:
            loops['hairpin'] += 1

    return loops


def predict_secondary_structure_simple(sequence: str) -> str:
    """
    Simple secondary structure prediction (placeholder).

    In a real implementation, this would use tools like:
    - ViennaRNA RNAfold
    - EternaFold
    - RNAstructure

    For now, returns a simple stem-loop structure as example.
    """
    length = len(sequence)

    if length < 6:
        return '.' * length

    # Simple heuristic: create a stem-loop structure
    # This is just for demonstration
    stem_length = min(4, length // 3)
    loop_length = max(3, length - 2 * stem_length)

    structure = '(' * stem_length + '.' * loop_length + ')' * stem_length

    # Pad or truncate to match sequence length
    if len(structure) < length:
        structure += '.' * (length - len(structure))
    elif len(structure) > length:
        structure = structure[:length]

    return structure


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_rna_structure_analysis(
    sequence: Optional[str] = None,
    secondary_structure: Optional[str] = None,
    predict_structure: bool = False,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze RNA secondary structure properties and statistics.

    Args:
        sequence: RNA sequence (for prediction)
        secondary_structure: Secondary structure in dot-bracket notation
        predict_structure: Whether to predict structure from sequence
        output_file: Path to save results (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - structure_analysis: Analysis results
            - validation: Structure validation results
            - config_used: Configuration used
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_rna_structure_analysis(
        ...     secondary_structure="((((....))))",
        ...     output_file="results/analysis.json"
        ... )
        >>> print(result['structure_analysis']['total_base_pairs'])
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate inputs
    if not sequence and not secondary_structure:
        raise ValueError("Must provide either sequence or secondary_structure")

    if predict_structure and not sequence:
        raise ValueError("Cannot predict structure without sequence")

    # Initialize results
    results = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "sequence": sequence,
            "secondary_structure": secondary_structure,
            "predict_structure": predict_structure
        },
        "structure_analysis": {},
        "validation": {},
        "prediction": {},
        "summary": {}
    }

    try:
        # Get or predict secondary structure
        if secondary_structure:
            target_structure = secondary_structure
            if config["verbose"]:
                print(f"📝 Analyzing provided structure: {target_structure}")
        elif predict_structure and sequence:
            target_structure = predict_secondary_structure_simple(sequence)
            results["prediction"] = {
                "method": "simple_heuristic",
                "predicted_structure": target_structure,
                "note": "This is a placeholder prediction. Use ViennaRNA or EternaFold for real predictions."
            }
            if config["verbose"]:
                print(f"🔮 Predicted structure: {target_structure}")
        else:
            raise ValueError("No structure provided and prediction not requested")

        # Validate structure
        if config["validate_structure"]:
            is_valid, validation_message = validate_dotbracket(target_structure)
            results["validation"] = {
                "is_valid": is_valid,
                "message": validation_message
            }

            if not is_valid:
                if config["verbose"]:
                    print(f"⚠️  Structure validation failed: {validation_message}")
            elif config["verbose"]:
                print("✅ Structure validation passed")

        # Analyze structure properties
        if config["include_statistics"]:
            analysis = analyze_structure_properties(target_structure)
            results["structure_analysis"] = analysis

            # Add sequence-specific analysis if available
            if sequence:
                results["structure_analysis"]["sequence_length"] = len(sequence)
                results["structure_analysis"]["length_match"] = len(sequence) == len(target_structure)

                # Simple GC content calculation
                if sequence:
                    gc_content = (sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence)
                    results["structure_analysis"]["gc_content"] = gc_content

        # Create summary
        if results["structure_analysis"]:
            analysis = results["structure_analysis"]
            results["summary"] = {
                "valid_dotbracket": results["validation"].get("is_valid", True),
                "length": analysis.get("length", 0),
                "paired_positions": analysis.get("paired_positions", 0),
                "unpaired_positions": analysis.get("unpaired_positions", 0),
                "pairing_percentage": analysis.get("pairing_percentage", 0),
                "total_base_pairs": analysis.get("total_base_pairs", 0),
                "has_pseudoknots": analysis.get("has_pseudoknots", False),
                "pseudoknot_order": analysis.get("pseudoknot_order", 0)
            }

        # Save output if requested
        output_path = None
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results as JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            if config["verbose"]:
                print(f"📁 Results saved to: {output_path}")

        # Add metadata
        results["metadata"] = {
            "structure": target_structure,
            "analysis_complete": True,
            "timestamp": datetime.now().isoformat()
        }

        return {
            **results,
            "config_used": config,
            "output_file": str(output_path) if output_path else None
        }

    except Exception as e:
        # Return error information for debugging
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "config_used": config,
            "metadata": {
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
    parser.add_argument('--sequence', help='RNA sequence')
    parser.add_argument('--secondary_structure', help='Secondary structure in dot-bracket notation')
    parser.add_argument('--predict', action='store_true', help='Predict structure from sequence')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--config', help='Config file (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Add CLI options to config
    if config is None:
        config = {}
    if args.verbose:
        config['verbose'] = True

    # Run analysis
    try:
        result = run_rna_structure_analysis(
            sequence=args.sequence,
            secondary_structure=args.secondary_structure,
            predict_structure=args.predict,
            output_file=args.output,
            config=config
        )

        if result.get('success', True):
            print(f"✅ Success: Structure analysis complete")
            if result.get('output_file'):
                print(f"📁 Output saved to: {result['output_file']}")

            # Print key results
            if 'summary' in result and result['summary']:
                summary = result['summary']
                print(f"\n📊 Analysis Summary:")
                print(f"   Length: {summary.get('length', 0)} nucleotides")
                print(f"   Base pairs: {summary.get('total_base_pairs', 0)}")
                print(f"   Pairing: {summary.get('pairing_percentage', 0):.1%}")
                print(f"   Pseudoknots: {'Yes' if summary.get('has_pseudoknots', False) else 'No'}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())