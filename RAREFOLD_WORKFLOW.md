# RareFold Workflow Documentation

## Overview
RareFold is a deep learning framework for:
1. **Structure Prediction**: Predicting protein structures containing noncanonical amino acids (NCAAs)
2. **Binder Design**: Designing peptide binders (linear or cyclic) with noncanonical amino acids using EvoBindRare

---

## Repository Structure

```
RareFold_envs/
├── src/
│   ├── predict_sc.py                      # Main prediction script
│   ├── make_msa_seq_feats.py             # MSA feature generation
│   ├── mc_design_length_var_batch.py     # Efficient batch design script
│   ├── mc_design_length_var_mt_batch.py  # Multi-threaded design variant
│   ├── convert_design_metrics_hr.py      # Convert metrics to human-readable format
│   └── rarefold/                         # Core package
│       ├── common/                       # Common utilities and constants
│       │   ├── protein.py               # Protein structure handling
│       │   └── residue_constants.py     # Amino acid definitions (49 types)
│       ├── data/                        # Data processing
│       │   ├── parsers.py              # MSA and FASTA parsers
│       │   └── pipeline.py             # Data pipeline
│       └── model/                       # Neural network model
│           ├── modules.py              # RareFold model architecture
│           ├── features.py             # Feature processing
│           ├── config.py               # Model configuration
│           ├── folding.py              # Structure generation
│           ├── all_atom.py             # Atom-level representations
│           └── tf/                     # TensorFlow utilities
├── predict.sh                           # Prediction workflow script
├── design_eff.sh                       # Efficient design workflow script
└── data/
    ├── params/                         # Model parameters
    │   ├── params20000.npy            # Prediction parameters
    │   └── finetuned_params25000.npy  # Design parameters
    └── uniclust30_2018_08/            # MSA database
```

---

## Workflow 1: Structure Prediction

### Shell Script: `predict.sh`

**Purpose**: Predict the 3D structure of a protein containing noncanonical amino acids

### Step-by-Step Workflow

```
Input: FASTA sequence with NCAAs
    ↓
Step 1: MSA Generation (HHblits)
    ↓
Step 2: MSA Feature Extraction
    ↓
Step 3: Structure Prediction
    ↓
Output: PDB structure file
```

### Detailed Steps

#### **Step 1: Multiple Sequence Alignment (MSA) Generation**
**Tool**: HHblits (external)
**Location**: `hh-suite/build/bin/hhblits`

**Input**:
- FASTA file with NCAAs replaced by 'X' (`1FU0_X.fasta`)
- Database: `uniclust30_2018_08`

**Command**:
```bash
hhblits -i $FASTAWITHX -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $MSA
```

**Output**:
- `.a3m` file (Multiple Sequence Alignment)

**Why**: MSA provides evolutionary information that helps the model understand protein structure constraints

---

#### **Step 2: MSA Feature Processing**
**Python File**: `src/make_msa_seq_feats.py`

**Key Functions**:
- `make_sequence_features()` (lines 21-35)
  - Creates one-hot encoding of sequence
  - Generates residue indices
  - Sets up sequence metadata

- `make_msa_features()` (lines 38-66)
  - Converts MSA to integer representation
  - Creates deletion matrices
  - Removes duplicate sequences

- `process()` (lines 70-104)
  - Main orchestrator function
  - Parses FASTA and MSA files
  - Combines sequence and MSA features

**Input**:
- FASTA file with 'X' for NCAAs
- MSA file (`.a3m` format)

**Output**:
- `msa_features.pkl` - Pickled dictionary containing:
  - `aatype`: One-hot encoded amino acid types
  - `between_segment_residues`: Segment boundaries
  - `domain_name`: Protein identifier
  - `residue_index`: Position indices
  - `seq_length`: Sequence length
  - `sequence`: Raw sequence string
  - `deletion_matrix_int`: Deletion information from MSA
  - `msa`: Integer-encoded MSA
  - `num_alignments`: Number of sequences in MSA

**Dependencies**:
- `rarefold.common.residue_constants`: AA definitions
- `rarefold.data.parsers`: MSA parsing utilities

---

#### **Step 3: Structure Prediction**
**Python File**: `src/predict_sc.py`

**Key Functions**:

1. `read_fasta()` (lines 48-58)
   - Reads FASTA file with NCAA three-letter codes
   - Example: `MEKKEF-SEP-IMGVM` (SEP is a noncanonical AA)

2. `get_int_seq()` (lines 133-164)
   - Maps NCAA three-letter codes to integer indices
   - Validates NCAA positions match MSA 'X' positions
   - Returns integer sequence representation

3. `make_features()` (lines 77-126)
   - Updates MSA features with NCAA information
   - Creates atom mappings (14-atom and 37-atom representations)
   - Generates target features and masks
   - Calls `process_features()` to sample MSA

4. `predict()` (lines 166-229)
   - **Main prediction function**
   - Defines forward pass through RareFold model (lines 185-194)
   - Loads pre-trained parameters
   - Runs JAX-based inference
   - Calculates pLDDT (predicted local distance difference test)

5. `save_structure()` (lines 233-255)
   - Converts predictions to PDB format
   - Adds pLDDT scores to B-factors
   - Saves final structure

**Model Architecture** (from `rarefold.model.modules`):
- `RareFold` class: Main model
  - Evoformer blocks (MSA and pair representations)
  - Structure module (generates 3D coordinates)
  - Recycling mechanism (iterative refinement)

**Input**:
- `msa_features.pkl`
- FASTA with NCAAs
- Model parameters (`params20000.npy`)
- Number of recycles (default: 3)

**Output**:
- `{ID}_pred.pdb`: Predicted structure with NCAAs in chain B

**Key Technologies**:
- JAX: Auto-differentiation and GPU acceleration
- Haiku: Neural network library
- TensorFlow: Feature processing

---

## Workflow 2: Peptide Binder Design (EvoBindRare)

### Shell Script: `design_eff.sh`

**Purpose**: Design novel peptide binders targeting a protein sequence

### Step-by-Step Workflow

```
Input: Target protein FASTA
    ↓
Step 1: MSA Generation (HHblits)
    ↓
Step 2: MSA Feature Extraction
    ↓
Step 3: Monte Carlo Design (Iterative)
    │   ├── Initialize random sequences
    │   ├── For each iteration:
    │   │   ├── Mutate sequences
    │   │   ├── Update features
    │   │   ├── Predict structure (GPU)
    │   │   ├── Calculate loss
    │   │   └── Select best sequences
    │   └── Repeat
    ↓
Step 4: Convert Metrics
    ↓
Output: Designed binders + metrics
```

### Detailed Steps

#### **Step 1 & 2: Same as Prediction Workflow**
Generate MSA and extract features for the **target protein**

---

#### **Step 3: Monte Carlo Design**
**Python File**: `src/mc_design_length_var_batch.py`

This is the core design engine with sophisticated parallel optimization.

**Design Parameters** (from `design_eff.sh`):
- `BINDER_LENGTHS`: "10,11,12,13,14,15" - Design multiple lengths simultaneously
- `BATCH_SIZE`: 5 - Number of parallel initializations per length
- `NITER`: 1000 - Number of optimization iterations
- `RESAMPLE_FREQ`: 100 - How often to resample MSA (avoid local minima)
- `MAX_RECYCLES`: 3 - Prediction recycles
- `RARE_AAS`: List of NCAAs to use (e.g., "MSE,MLY,PTR,SEP")
- `CYCLIC`: True/False - Enable cyclic peptide design
- `MAX_WORKERS`: 30 - CPU threads for parallelization

**Key Functions**:

1. **Parallelization** (lines 108-150)
   - `parallel_map()`: Distributes work across CPU cores
   - Uses `ProcessPoolExecutor` for true multiprocessing
   - Dramatically reduces CPU overhead

2. **Feature Initialization** (lines 218-275)
   - `init_features()`: Creates initial feature dictionaries for binders
   - Concatenates binder sequence to target protein
   - Updates MSA, deletion matrices, residue indices
   - Handles cyclic offset arrays (for cyclic peptides)

3. **Batch Uniformization** (lines 277-340)
   - `uniform_batch()`: Pads sequences to same length for efficient GPU batching
   - Creates uniform tensors for all features
   - Key insight: Different length binders in same batch!

4. **Sequence Initialization** (lines 343-359)
   - `initialize_weights()`: Random Gumbel sampling
   - Initializes diverse starting sequences
   - Supports custom amino acid alphabet (20 regular + NCAAs)

5. **Mutation** (lines 362-387)
   - `mutate_sequence()`: Monte Carlo mutation
   - Random position and amino acid selection
   - Checks against previously searched sequences
   - Explores sequence space systematically

6. **Feature Update** (lines 453-490)
   - `update_peptide_batch_feats()`: Fast feature updates
   - Only updates changed parts (binder sequence)
   - Uses TensorFlow gather operations for efficiency
   - Updates atom mappings for NCAAs

7. **Loss Calculation** (lines 492-554)
   - `get_loss()`: Multi-objective optimization
   - **Components**:
     - `if_dist_binder`: Interface distance (closer = better binding)
     - `plddt`: Predicted local distance difference test (higher = better confidence)
     - `inter_clash_frac`: Inter-molecular clashes (lower = better)
     - `intra_clash_frac`: Intra-molecular clashes (lower = better)
   - **Combined Loss**: `if_dist * 1/plddt + inter_clash + intra_clash`

8. **Main Design Loop** (`design_binder()`, lines 558-859)

**Initialization Phase** (lines 676-750):
```python
for each binder_length in [10,11,12,13,14,15]:
    for each init in [1,2,3,4,5]:  # batch_size
        - Initialize random sequence
        - Create features
        - Predict structure
        - Calculate loss
        - Save init structure
```

**Optimization Phase** (lines 760-859):
```python
for iteration in range(1000):
    # Parallel mutation (CPU)
    mutated_sequences = parallel_map(mutate_sequence, ...)

    # Feature update
    if iteration % 100 == 0:
        # Resample MSA (avoid local minima)
        batch = resample_msa_and_rebuild()
    else:
        # Fast update (only binder features)
        batch = update_peptide_batch_feats()

    # Prediction (GPU) - all designs in parallel!
    prediction_result = vmap_apply_fwd(params, rng, batch)

    # Parallel loss calculation (CPU)
    losses = parallel_map(get_loss, ...)

    # Parallel structure saving (CPU)
    parallel_map(save_structure, ...)

    # Selection
    best_sequences = argmin(losses)

    # Continue from best
    current_sequences = best_sequences
```

**Efficiency Innovations**:
- **Batch consolidation**: All 30 designs (6 lengths × 5 inits) on 1 GPU instead of 30 GPUs
- **Parallel CPU processing**: Mutation, feature updates, loss calculations all parallelized
- **GPU off-time reduction**: CPU overhead reduced from 600s to 20s per iteration!
- **Smart MSA resampling**: Every 100 iterations to escape local minima

**Output**:
- `metrics.csv`: Raw metrics for all iterations
- Structure files organized by binder length:
  - `{outdir}/10/iter_500_0.pdb`
  - `{outdir}/11/iter_500_0.pdb`
  - etc.

---

#### **Step 4: Metrics Conversion**
**Python File**: `src/convert_design_metrics_hr.py`

**Purpose**: Convert batch-formatted metrics to human-readable row-wise format

**Input**:
- `metrics.csv` (batch format)

**Processing** (lines 24-48):
- Parses literal Python lists from CSV
- Transposes data from column-wise to row-wise
- Creates separate row for each replicate and length

**Output**:
- `metrics_hr.csv` with columns:
  - `iter`: Iteration number
  - `replicate`: Which initialization (0-4)
  - `length`: Binder length (10-15)
  - `if_dist_binder`: Interface distance
  - `plddt`: Confidence score
  - `inter_clash_frac`: Inter-molecular clashes
  - `intra_clash_frac`: Intra-molecular clashes
  - `loss`: Combined loss
  - `sequence`: Three-letter code sequence
  - `int_seq`: Integer representation

---

## Core Model Components

### `rarefold/common/residue_constants.py`
- Defines 49 amino acid types (20 regular + 29 NCAAs)
- Atom type definitions for each residue
- Mapping between 14-atom and 37-atom representations

### `rarefold/model/modules.py`
- **RareFold** class: Main neural network
- Evoformer: Processes MSA and generates pair representations
- Structure Module: Generates 3D coordinates from representations
- Cyclic offset support: Special handling for cyclic peptides

### `rarefold/model/features.py`
- `np_example_to_features()`: Converts raw features to model inputs
- MSA sampling and clustering
- Feature normalization

### `rarefold/model/folding.py`
- Structure generation module
- Converts abstract representations to 3D atom coordinates
- Uses quaternion-based rigid body transformations

---

## Data Flow Summary

### Prediction Flow:
```
FASTA → HHblits → MSA (.a3m)
                     ↓
              make_msa_seq_feats.py
                     ↓
              msa_features.pkl
                     ↓
              predict_sc.py
              ├── Read FASTA (with NCAAs)
              ├── Map NCAAs to integers
              ├── Create features
              ├── Load model params
              ├── JAX prediction
              └── Save PDB
                     ↓
              {ID}_pred.pdb
```

### Design Flow:
```
Target FASTA → HHblits → MSA (.a3m)
                             ↓
                    make_msa_seq_feats.py
                             ↓
                       msa_features.pkl
                             ↓
                    mc_design_length_var_batch.py
                    ├── Initialize sequences (random)
                    ├── For each iteration:
                    │   ├── Mutate (parallel CPU)
                    │   ├── Update features
                    │   ├── Predict (batched GPU)
                    │   ├── Calculate loss (parallel CPU)
                    │   ├── Save structures (parallel CPU)
                    │   └── Select best
                    └── Output metrics.csv
                             ↓
                    convert_design_metrics_hr.py
                             ↓
                       metrics_hr.csv + PDB files
```

---

## Key Technologies

1. **JAX**: Auto-differentiation, GPU acceleration, functional programming
2. **Haiku**: Neural network library (built on JAX)
3. **TensorFlow**: Feature processing utilities
4. **HHblits**: MSA search tool
5. **NumPy**: Numerical operations
6. **Pandas**: Data management
7. **Concurrent.futures**: Parallel processing

---

## Time Efficiency (Design)

| Component | Old (30 GPUs) | New (1 GPU) | Improvement |
|-----------|---------------|-------------|-------------|
| GPU Hardware | 30x GPUs | 1x GPU | **30x reduction** |
| Prediction | 91.29s | 91.29s | Same |
| CPU Overhead | 600.60s | 20.02s | **30x faster** |
| **Total per iteration** | ~692s | ~111s | **6x faster** |

**CPU Overhead Breakdown** (New):
- Mutating sequences: 3.16s
- Loss calculations: 3.41s
- Making new features: 0.04s
- Adding metrics: 0.01s
- Saving structures: 13.40s
- **Total**: 20.02s

---

## Supported Noncanonical Amino Acids (29 total)

MSE, TPO, MLY, CME, PTR, SEP, SAH, CSO, PCA, KCX, CAS, CSD, MLZ, OCS, ALY, CSS, CSX, HIC, HYP, YCM, YOF, M3L, PFF, CGU, FTR, LLP, CAF, CMH, MHO

---

## Citation
Li Q, Daumiller D, Zuo F, Marcotte H, Pan-Hammarstrom Q and Bryant P. RareFold: Structure prediction and design of proteins with noncanonical amino acids. bioRxiv. 2025.
