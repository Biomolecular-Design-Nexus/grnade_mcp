"""MCP Server for gRNAde

Provides both synchronous and asynchronous (submit) APIs for RNA structure analysis,
evaluation, inverse design, and batch processing.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager

# Create MCP server
mcp = FastMCP("grnade")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def analyze_rna_structure(
    secondary_structure: Optional[str] = None,
    sequence: Optional[str] = None,
    predict_structure: bool = False,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Analyze RNA secondary structure properties and statistics.

    Fast operation suitable for single sequences (~30 seconds).
    Fully independent tool that validates dot-bracket notation and calculates
    structure statistics including base pairs, stems, loops, and pseudoknots.

    Args:
        secondary_structure: Secondary structure in dot-bracket notation
        sequence: RNA sequence (for structure prediction)
        predict_structure: Whether to predict structure from sequence
        output_file: Optional path to save results as JSON
        verbose: Include detailed analysis output

    Returns:
        Dictionary with structure analysis results, validation, and statistics

    Example:
        analyze_rna_structure(secondary_structure="(((...)))")
        analyze_rna_structure(sequence="GGGAAACCC", predict_structure=True)
    """
    # Import the script's main function
    try:
        from rna_structure_analysis import run_rna_structure_analysis

        result = run_rna_structure_analysis(
            secondary_structure=secondary_structure,
            sequence=sequence,
            predict_structure=predict_structure,
            output_file=output_file,
            verbose=verbose
        )
        return {"status": "success", **result}
    except ImportError as e:
        return {"status": "error", "error": f"Failed to import script: {e}"}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Analysis failed: {str(e)}"}


@mcp.tool()
def evaluate_rna_sequences(
    sequences: Union[List[str], str],
    target_structure: str,
    output_file: Optional[str] = None,
    use_basic_stats: bool = True,
    verbose: bool = True
) -> dict:
    """
    Evaluate RNA sequences using computational metrics.

    Fast operation suitable for small sequence sets (~1-2 minutes for <100 sequences).
    Uses graceful fallback to basic statistics when advanced models are unavailable.

    Args:
        sequences: List of RNA sequences or comma-separated string
        target_structure: Target secondary structure in dot-bracket notation
        output_file: Optional path to save results as CSV
        use_basic_stats: Whether to use basic statistics mode
        verbose: Include detailed evaluation output

    Returns:
        Dictionary with evaluation results and summary statistics

    Example:
        evaluate_rna_sequences(["GGGAAACCC", "AUCGAUCG"], "(((...)))")
        evaluate_rna_sequences("GGGAAACCC,AUCGAUCG", "(((...)))")
    """
    try:
        from rna_evaluation import run_rna_evaluation

        # Handle comma-separated string input
        if isinstance(sequences, str):
            sequences = [s.strip() for s in sequences.split(",")]

        result = run_rna_evaluation(
            sequences=sequences,
            target_structure=target_structure,
            output_file=output_file,
            use_basic_stats=use_basic_stats,
            verbose=verbose
        )
        return {"status": "success", **result}
    except ImportError as e:
        return {"status": "error", "error": f"Failed to import script: {e}"}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Evaluation failed: {str(e)}"}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_rna_inverse_design(
    secondary_structure: Optional[str] = None,
    pdb_file: Optional[str] = None,
    mode: str = "2d",
    n_designs: int = 10,
    partial_seq: Optional[str] = None,
    temperature_min: float = 0.1,
    temperature_max: float = 1.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit RNA inverse design for background processing.

    This operation generates RNA sequences that fold into specified 2D/3D structures
    using gRNAde models. May take >10 minutes depending on design complexity.

    Args:
        secondary_structure: Secondary structure in dot-bracket notation (2D mode)
        pdb_file: Path to PDB file (3D mode)
        mode: Design mode - "2d" or "3d"
        n_designs: Number of sequences to generate (default: 10)
        partial_seq: Partial sequence constraints (optional)
        temperature_min: Minimum sampling temperature (default: 0.1)
        temperature_max: Maximum sampling temperature (default: 1.0)
        output_dir: Directory to save outputs
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        submit_rna_inverse_design(secondary_structure="(((...)))", mode="2d", n_designs=20)
        submit_rna_inverse_design(pdb_file="structure.pdb", mode="3d", n_designs=50)
    """
    script_path = str(SCRIPTS_DIR / "rna_inverse_design.py")

    args = {
        "mode": mode,
        "n_designs": n_designs,
        "temperature_min": temperature_min,
        "temperature_max": temperature_max
    }

    if secondary_structure:
        args["secondary_structure"] = secondary_structure
    if pdb_file:
        args["pdb_file"] = pdb_file
    if partial_seq:
        args["partial_seq"] = partial_seq
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"rna_design_{mode}_{n_designs}_sequences"
    )


@mcp.tool()
def submit_batch_rna_pipeline(
    targets_file: Optional[str] = None,
    pdb_dir: Optional[str] = None,
    targets: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    n_designs_per_target: int = 50,
    max_workers: Optional[int] = None,
    enable_evaluation: bool = True,
    enable_filtering: bool = True,
    max_results_per_target: int = 10,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA design pipeline for multiple targets.

    This operation runs high-throughput RNA design for multiple targets with
    evaluation and filtering. Can take >30 minutes for large datasets.

    Args:
        targets_file: Path to CSV file with target specifications
        pdb_dir: Directory containing PDB files to process
        targets: List of target dictionaries (alternative to files)
        output_dir: Directory to save all outputs
        n_designs_per_target: Number of sequences to generate per target (default: 50)
        max_workers: Maximum number of parallel workers (default: auto)
        enable_evaluation: Whether to run evaluation phase (default: True)
        enable_filtering: Whether to run filtering phase (default: True)
        max_results_per_target: Maximum results to keep per target (default: 10)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the batch pipeline

    Example:
        submit_batch_rna_pipeline(targets_file="targets.csv", output_dir="results/batch")
        submit_batch_rna_pipeline(pdb_dir="structures/", n_designs_per_target=100)
    """
    script_path = str(SCRIPTS_DIR / "batch_rna_pipeline.py")

    args = {
        "n_designs_per_target": n_designs_per_target,
        "enable_evaluation": enable_evaluation,
        "enable_filtering": enable_filtering,
        "max_results_per_target": max_results_per_target
    }

    if targets_file:
        args["targets_file"] = targets_file
    elif pdb_dir:
        args["pdb_dir"] = pdb_dir
    elif targets:
        # Save targets to temporary file
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(targets, f)
            args["targets_file"] = f.name

    if output_dir:
        args["output_dir"] = output_dir
    if max_workers:
        args["max_workers"] = max_workers

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_pipeline_{n_designs_per_target}_per_target"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_rna_evaluation(
    sequences_list: List[List[str]],
    target_structures: List[str],
    sequence_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    use_basic_stats: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA sequence evaluation for multiple sequence sets.

    Processes multiple sets of sequences with their target structures in a single job.
    Suitable for evaluating sequences from multiple design runs.

    Args:
        sequences_list: List of sequence lists (one list per target)
        target_structures: List of target structures (one per sequence list)
        sequence_names: Optional names for each sequence set
        output_dir: Directory to save all outputs
        use_basic_stats: Whether to use basic statistics mode
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch evaluation

    Example:
        submit_batch_rna_evaluation(
            sequences_list=[["GGGAAACCC", "AUCGAUCG"], ["CCCCAAAAGGG"]],
            target_structures=["(((...)))", "((((...))))"]
        )
    """
    # This is implemented by running multiple evaluation jobs
    import tempfile
    import json

    # Create temporary input file
    batch_data = {
        "sequences_list": sequences_list,
        "target_structures": target_structures,
        "sequence_names": sequence_names or [f"set_{i}" for i in range(len(sequences_list))]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(batch_data, f)
        batch_file = f.name

    script_path = str(SCRIPTS_DIR / "rna_evaluation.py")

    args = {
        "batch_file": batch_file,
        "use_basic_stats": use_basic_stats
    }

    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_eval_{len(sequences_list)}_targets"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_rna_inputs(
    sequence: Optional[str] = None,
    secondary_structure: Optional[str] = None,
    pdb_file: Optional[str] = None
) -> dict:
    """
    Validate RNA inputs before processing.

    Args:
        sequence: RNA sequence to validate
        secondary_structure: Secondary structure to validate
        pdb_file: PDB file path to validate

    Returns:
        Dictionary with validation results for each input
    """
    results = {"status": "success", "validation_results": {}}

    try:
        # Validate sequence
        if sequence:
            valid_bases = set("AUGC")
            sequence = sequence.upper().replace("T", "U")  # Convert DNA to RNA
            invalid_bases = set(sequence) - valid_bases

            if invalid_bases:
                results["validation_results"]["sequence"] = {
                    "valid": False,
                    "error": f"Invalid bases found: {invalid_bases}"
                }
            else:
                results["validation_results"]["sequence"] = {
                    "valid": True,
                    "length": len(sequence),
                    "gc_content": (sequence.count("G") + sequence.count("C")) / len(sequence)
                }

        # Validate secondary structure
        if secondary_structure:
            from rna_structure_analysis import validate_dotbracket
            is_valid, message = validate_dotbracket(secondary_structure)
            results["validation_results"]["secondary_structure"] = {
                "valid": is_valid,
                "message": message,
                "length": len(secondary_structure)
            }

        # Validate PDB file
        if pdb_file:
            pdb_path = Path(pdb_file)
            if pdb_path.exists():
                try:
                    with open(pdb_path, 'r') as f:
                        lines = f.readlines()

                    atom_count = sum(1 for line in lines if line.startswith("ATOM"))
                    results["validation_results"]["pdb_file"] = {
                        "valid": True,
                        "exists": True,
                        "atom_count": atom_count,
                        "file_size": pdb_path.stat().st_size
                    }
                except Exception as e:
                    results["validation_results"]["pdb_file"] = {
                        "valid": False,
                        "exists": True,
                        "error": f"Failed to read PDB file: {e}"
                    }
            else:
                results["validation_results"]["pdb_file"] = {
                    "valid": False,
                    "exists": False,
                    "error": "PDB file does not exist"
                }

    except Exception as e:
        return {"status": "error", "error": f"Validation failed: {e}"}

    return results


@mcp.tool()
def get_example_data() -> dict:
    """
    Get information about available example datasets for testing.

    Returns:
        Dictionary with example files, descriptions, and usage examples
    """
    examples = {
        "status": "success",
        "example_data": {
            "sequences": {
                "simple_hairpin": "GGGAAACCC",
                "complex_hairpin": "GGGGAAAACCCC",
                "longer_structure": "GGGCCCAAAUUUGGGCCC",
                "description": "Example RNA sequences for testing"
            },
            "secondary_structures": {
                "simple_hairpin": "(((...)))",
                "complex_hairpin": "((((....))))",
                "longer_structure": "(((....)))..(((..)))",
                "pseudoknot": "((([[[.)))]]]",
                "description": "Example secondary structures in dot-bracket notation"
            },
            "example_commands": {
                "structure_analysis": [
                    'analyze_rna_structure(secondary_structure="(((...)))")',
                    'analyze_rna_structure(sequence="GGGAAACCC", predict_structure=True)'
                ],
                "sequence_evaluation": [
                    'evaluate_rna_sequences(["GGGAAACCC", "AUCGAUCG"], "(((...)))")'
                ],
                "rna_design": [
                    'submit_rna_inverse_design(secondary_structure="(((...)))", mode="2d", n_designs=20)'
                ]
            }
        }
    }

    return examples


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()