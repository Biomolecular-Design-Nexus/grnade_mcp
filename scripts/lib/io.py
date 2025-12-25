"""
Shared I/O functions for MCP scripts.

These functions provide a common interface for loading and saving various file formats
used in RNA analysis, extracted and simplified from repo code to minimize dependencies.
"""

import json
import csv
from pathlib import Path
from typing import Union, Dict, List, Any, Optional
import pandas as pd


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file with proper formatting."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_sequences_from_csv(file_path: Union[str, Path], sequence_column: Optional[str] = None) -> List[str]:
    """
    Load RNA sequences from CSV file.

    Args:
        file_path: Path to CSV file
        sequence_column: Name of column containing sequences (auto-detect if None)

    Returns:
        List of sequence strings
    """
    df = pd.read_csv(file_path)

    # Auto-detect sequence column if not specified
    if sequence_column is None:
        possible_names = ['sequence', 'seq', 'rna_sequence', 'sequences']
        for name in possible_names:
            if name in df.columns:
                sequence_column = name
                break
        else:
            # Use first column as fallback
            sequence_column = df.columns[0]

    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in {file_path}")

    return df[sequence_column].dropna().tolist()


def save_sequences_to_csv(sequences: List[str], file_path: Union[str, Path],
                         additional_data: Optional[Dict[str, List]] = None) -> None:
    """
    Save sequences to CSV file with optional additional columns.

    Args:
        sequences: List of sequence strings
        file_path: Output file path
        additional_data: Dict of column_name -> list_of_values for extra columns
    """
    data = {'sequence': sequences}

    if additional_data:
        # Validate all lists have same length
        for key, values in additional_data.items():
            if len(values) != len(sequences):
                raise ValueError(f"Length mismatch: {len(values)} vs {len(sequences)} for column '{key}'")
            data[key] = values

    df = pd.DataFrame(data)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def load_fasta_sequences(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load sequences from FASTA file.

    Returns:
        Dict of {header: sequence}
    """
    sequences = {}
    current_header = None
    current_sequence = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_header and current_sequence:
                    sequences[current_header] = ''.join(current_sequence)

                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_sequence = []
            elif current_header and line:
                current_sequence.append(line)

    # Save last sequence
    if current_header and current_sequence:
        sequences[current_header] = ''.join(current_sequence)

    return sequences


def save_fasta_sequences(sequences: Dict[str, str], file_path: Union[str, Path]) -> None:
    """
    Save sequences to FASTA file.

    Args:
        sequences: Dict of {header: sequence}
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for header, sequence in sequences.items():
            f.write(f">{header}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i:i+80]}\n")


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """Validate that file exists and return Path object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def create_output_directory(output_path: Union[str, Path]) -> Path:
    """Create output directory and return Path object."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension in lowercase."""
    return Path(file_path).suffix.lower()


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    extension = get_file_extension(config_path)

    if extension == '.json':
        return load_json(config_path)
    elif extension in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    else:
        raise ValueError(f"Unsupported config file format: {extension}")


def save_dataframe_multiple_formats(df: pd.DataFrame, base_path: Union[str, Path],
                                   formats: List[str] = ["csv"]) -> List[str]:
    """
    Save DataFrame in multiple formats.

    Args:
        df: DataFrame to save
        base_path: Base path without extension
        formats: List of formats to save ("csv", "json", "excel")

    Returns:
        List of saved file paths
    """
    base_path = Path(base_path)
    saved_files = []

    for fmt in formats:
        if fmt == "csv":
            file_path = base_path.with_suffix('.csv')
            df.to_csv(file_path, index=False)
        elif fmt == "json":
            file_path = base_path.with_suffix('.json')
            df.to_json(file_path, orient='records', indent=2)
        elif fmt == "excel":
            file_path = base_path.with_suffix('.xlsx')
            df.to_excel(file_path, index=False)
        else:
            continue

        saved_files.append(str(file_path))

    return saved_files


def load_targets_from_various_formats(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load target specifications from various file formats.

    Supports CSV, JSON, and directory of PDB files.

    Returns:
        List of target dictionaries
    """
    input_path = Path(input_path)

    if input_path.is_file():
        extension = get_file_extension(input_path)

        if extension == '.csv':
            df = pd.read_csv(input_path)
            return df.to_dict('records')
        elif extension == '.json':
            data = load_json(input_path)
            if isinstance(data, list):
                return data
            else:
                return [data]
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    elif input_path.is_dir():
        # Load from directory of PDB files
        pdb_files = list(input_path.glob("*.pdb"))
        targets = []
        for pdb_file in pdb_files:
            targets.append({
                'name': pdb_file.stem,
                'pdb_file': str(pdb_file),
                'mode': '3d'
            })
        return targets

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")