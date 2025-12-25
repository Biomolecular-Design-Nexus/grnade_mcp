# MCP Scripts - Clean RNA Analysis Tools

Clean, self-contained scripts extracted from gRNAde use cases for MCP tool wrapping.

## Overview

These scripts provide RNA analysis capabilities with minimal dependencies and graceful fallback behavior. Each script is designed to work independently and can be easily wrapped as MCP tools.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Clean main functions ready for MCP wrapping
5. **Graceful Fallbacks**: Basic functionality when advanced models unavailable

## Quick Start

### Environment Setup

```bash
# Activate the environment (prefer mamba over conda)
mamba activate ../env  # or: conda activate ../env

# Install additional dependencies if needed
pip install pandas numpy torch
```

### Test Basic Functionality

```bash
# Test structure analysis (fully independent)
python rna_structure_analysis.py --secondary_structure "(((...)))" --verbose

# Test evaluation with basic stats (graceful fallback)
python rna_evaluation.py --sequences "GGGAAACCC,AUCGAUC" --target_structure "(((...)))" --verbose
```

## Scripts

### 🟢 rna_structure_analysis.py - RNA Secondary Structure Analysis
**Status**: ✅ Fully Independent | **Dependencies**: None (beyond standard packages)

Analyze RNA secondary structure properties, validate dot-bracket notation, and calculate comprehensive statistics.

```bash
# Basic structure analysis
python rna_structure_analysis.py --secondary_structure "(((...)))" --output results/analysis.json

# Predict structure from sequence
python rna_structure_analysis.py --sequence "GGGAAACCC" --predict --verbose

# With custom config
python rna_structure_analysis.py --secondary_structure "(((...)))" --config ../configs/rna_structure_analysis_config.json
```

**Features:**
- ✅ Dot-bracket notation validation
- ✅ Base pair identification and statistics
- ✅ Pseudoknot detection
- ✅ Stem and loop analysis
- ✅ Simple structure prediction (heuristic)
- ✅ JSON output with comprehensive metadata

**Example Output:**
```json
{
  "structure_analysis": {
    "length": 9,
    "total_base_pairs": 3,
    "pairing_percentage": 0.67,
    "has_pseudoknots": false,
    "num_stems": 1,
    "avg_stem_length": 3.0
  },
  "validation": {
    "is_valid": true,
    "message": "Valid dot-bracket notation"
  }
}
```

---

### 🟡 rna_evaluation.py - RNA Sequence Evaluation
**Status**: ⚠️ Graceful Fallback | **Dependencies**: RibonanzaNet models (optional)

Evaluate RNA sequences using computational metrics. Falls back to basic sequence statistics when advanced models are unavailable.

```bash
# Basic evaluation (works without models)
python rna_evaluation.py --sequences "GGGGAAAACCCC,AUCGAUC" --target_structure "(((...)))" --output eval.csv

# Evaluate from CSV file
python rna_evaluation.py --sequences_file sequences.csv --target_structure "(((...)))" --verbose

# With custom config
python rna_evaluation.py --sequences_file sequences.csv --target_structure "(((...)))" --config ../configs/rna_evaluation_config.json
```

**Features:**
- ✅ Basic sequence statistics (length, GC content, nucleotide composition)
- ⚠️ OpenKnot scoring (requires models)
- ⚠️ RibonanzaNet self-consistency scoring (requires models)
- ✅ CSV output with results
- ✅ Batch sequence processing
- ✅ Graceful degradation when models unavailable

**Fallback Mode**: When models are not available, returns basic sequence statistics:
```
sequence,length,gc_content,openknot_score,sc_score_ribonanzanet,sc_score_ribonanzanet_ss
GGGAAACCC,9,0.556,0.0,0.0,0.0
```

---

### 🔴 rna_inverse_design.py - RNA Sequence Generation
**Status**: ❌ Requires Models | **Dependencies**: gRNAde models and featurizers

Generate RNA sequences that fold into specified 2D or 3D structures using gRNAde.

```bash
# 2D mode (secondary structure input)
python rna_inverse_design.py --secondary_structure "(((...)))" --mode 2d --output_dir results/design

# 3D mode (PDB file input)
python rna_inverse_design.py --pdb structure.pdb --mode 3d --output_dir results/design

# With custom parameters
python rna_inverse_design.py --secondary_structure "(((...)))" --n_pass 50 --total_samples 500 --output_dir results/
```

**Features:**
- ⚠️ 2D mode (secondary structure → sequences)
- ⚠️ 3D mode (PDB structure → sequences)
- ⚠️ Partial sequence constraints
- ⚠️ Temperature-based sampling
- ✅ Configurable generation parameters
- ✅ CSV output with perplexity scores

**Requirements**: This script requires the full gRNAde setup with models downloaded.

---

### 🔴 batch_rna_pipeline.py - High-Throughput Batch Pipeline
**Status**: ❌ Requires Models | **Dependencies**: Working sub-scripts + models

High-throughput RNA design pipeline for multiple targets with evaluation and filtering.

```bash
# Process targets from CSV
python batch_rna_pipeline.py --targets_file targets.csv --output_dir results/batch --max_workers 4

# Process PDB directory
python batch_rna_pipeline.py --pdb_dir structures/ --output_dir results/batch

# Disable evaluation/filtering phases
python batch_rna_pipeline.py --targets_file targets.csv --output_dir results/ --no_evaluation --no_filtering
```

**Pipeline Phases:**
1. **Design Generation**: Generate RNA sequences for each target
2. **Evaluation** (optional): Score sequences using evaluation metrics
3. **Filtering & Ranking** (optional): Filter by quality and rank by scores

**Target CSV Format:**
```csv
target_name,pdb_file,secondary_structure,mode,description
hairpin1,,(((...))),,2d,Simple hairpin
protein1,structure.pdb,,3d,Protein binding site
```

---

## Configuration Files

All scripts support JSON configuration files in `../configs/`:

- `rna_structure_analysis_config.json` - Structure analysis settings
- `rna_evaluation_config.json` - Evaluation model paths and settings
- `rna_inverse_design_config.json` - Design generation parameters
- `batch_rna_pipeline_config.json` - Batch processing settings

**Example config usage:**
```bash
python rna_structure_analysis.py --secondary_structure "(((...)))" --config ../configs/rna_structure_analysis_config.json
```

## Shared Library

Common functionality is provided in `lib/`:

- `lib/constants.py` - RNA nucleotide mappings, thresholds, defaults
- `lib/io.py` - File loading/saving utilities (JSON, CSV, FASTA)
- `lib/utils.py` - General utilities (validation, filtering, diversity metrics)

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped as MCP tools:

```python
# Import the main function
from scripts.rna_structure_analysis import run_rna_structure_analysis

# Wrap as MCP tool
@mcp.tool()
def analyze_rna_structure(secondary_structure: str, output_file: str = None):
    """Analyze RNA secondary structure properties."""
    return run_rna_structure_analysis(secondary_structure=secondary_structure, output_file=output_file)
```

### Fast vs Background Operations

**Fast Operations** (suitable for direct MCP calls):
- `run_rna_structure_analysis()` - Structure analysis (~1 second)
- `run_rna_evaluation()` - Basic evaluation for small datasets (<30 seconds)

**Background Operations** (submit for async processing):
- `run_rna_inverse_design()` - Sequence generation (requires models)
- `run_batch_rna_pipeline()` - Batch processing for multiple targets

## Model Setup (for Advanced Features)

To enable advanced features that require models:

1. **Download Models**: Ensure gRNAde and RibonanzaNet models are available in:
   - `../models/`
   - `../examples/data/`
   - `../repo/geometric-rna-design/checkpoints/`

2. **Required Model Files**:
   - `gRNAde_drop3d@0.75_maxlen@500.h5` (main gRNAde model)
   - `ribonanzanet.pt` (SHAPE reactivity prediction)
   - `ribonanzanet_ss.pt` (secondary structure prediction)

3. **Environment Setup**: Ensure all gRNAde dependencies are installed in the conda/mamba environment

## Troubleshooting

### "No module named 'src'" or similar import errors
- This is expected when models are not available
- Scripts will fall back to basic functionality
- Check that repo is at `../repo/geometric-rna-design/`

### "Model checkpoint not found"
- Download required models to one of the search paths
- Check config files for correct model paths
- Use basic mode functionality instead

### "CUDA not available" warnings
- Scripts will automatically fall back to CPU
- For better performance, ensure CUDA is set up correctly
- Or explicitly set `device: "cpu"` in config files

## Dependencies

### Essential (required for all scripts):
- `python >= 3.8`
- `numpy`
- `pandas`
- `torch`
- Standard library packages

### Optional (for advanced features):
- Complete gRNAde environment setup
- PyTorch Geometric
- RibonanzaNet models
- gRNAde model checkpoints

## Performance Notes

- **Structure Analysis**: Very fast (<1 second), no external dependencies
- **Basic Evaluation**: Fast (<30 seconds for 100 sequences)
- **Advanced Evaluation**: Moderate (1-5 minutes with models)
- **RNA Design**: Slow (1-10 minutes depending on parameters and models)
- **Batch Pipeline**: Variable (scales with number of targets and parameters)

## Example Workflows

### Quick Analysis Workflow
```bash
# 1. Analyze structure
python rna_structure_analysis.py --secondary_structure "(((...)))" --output structure.json --verbose

# 2. Basic sequence evaluation
python rna_evaluation.py --sequences "GGGAAACCC,AUCGAUCGAUC" --target_structure "(((...)))" --output eval.csv --verbose
```

### Full Pipeline Workflow (requires models)
```bash
# 1. Create targets file
echo "target_name,secondary_structure,mode" > targets.csv
echo "hairpin1,(((...))),,2d" >> targets.csv

# 2. Run batch pipeline
python batch_rna_pipeline.py --targets_file targets.csv --output_dir results/batch --verbose

# 3. Check results
ls results/batch/
```

## Integration with MCP Server

These scripts are designed to be easily integrated into MCP servers. See the main project documentation for examples of wrapping these functions as MCP tools.

**Key Integration Points:**
- Clean function signatures with type hints
- Standardized return formats (dicts with success/error status)
- JSON-serializable inputs and outputs
- Comprehensive error handling
- Progress reporting capabilities

For more detailed information, see `../reports/step5_scripts.md`.