# RareFold and AlphaFold: Technical Relationship

## Executive Summary

**Yes, RareFold is built on top of AlphaFold's architecture.** It extends AlphaFold to handle noncanonical amino acids (NCAAs) by expanding the amino acid vocabulary from 20 to 49 types and extending the atom representation system. The core neural network architecture (Evoformer + Structure Module) remains the same, but the parameters are fine-tuned for NCAA-containing proteins.

---

## Evidence of AlphaFold Foundation

### 1. **Copyright and Attribution**

From `src/rarefold/common/residue_constants.py:1-15`:
```python
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ...
"""Constants used in AlphaFold."""
```

- **DeepMind copyright** indicates original AlphaFold codebase
- **Line 15**: Explicitly states "Constants used in AlphaFold"

### 2. **Code References to AlphaFold Paper**

Throughout the code, there are explicit references to:
- **Jumper et al. (2021)** - The AlphaFold 2 Nature paper
- Example from `src/rarefold/model/modules.py:276`:
  ```python
  class RareFold(hk.Module):
    """rarefold model with recycling.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference"
    """
  ```

### 3. **Parameter File Renaming**

From `src/predict_sc.py:204-210`:
```python
#Load params (need to do this here - need to enable GPU through jax first)
params = np.load(params , allow_pickle=True)
#Fix naming - tha params are saved using an old naming (alphafold)
new_params = {}
for key in params:
    new_key = re.sub('alphafold', 'rarefold', key)
    new_params[new_key] = params[key]
params = new_params
```

**This is smoking gun evidence**: The parameters were originally saved with "alphafold" naming and are renamed to "rarefold" at runtime.

---

## How RareFold Extends AlphaFold for Noncanonical Amino Acids

### The Challenge

AlphaFold was designed for the **20 canonical amino acids**. Each amino acid type has:
- A specific set of atoms (different for each AA)
- Specific chi angles (side-chain torsions)
- Specific 3D rigid body transformations

**Problem**: Noncanonical amino acids have:
- Different atoms (e.g., Se instead of S in MSE)
- Different chemical properties
- Modified side chains (e.g., phosphorylation adds PO3 group)

**RareFold's Solution**: Extend the entire system to support 49 amino acid types (20 canonical + 29 noncanonical).

---

## Key Modifications to AlphaFold

### 1. **Extended Amino Acid Vocabulary (20 → 49)**

#### **Original AlphaFold**:
```python
# 20 canonical amino acids
restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
restype_num = 20
```

#### **RareFold**:
From `src/rarefold/common/residue_constants.py:1171`:
```python
# Modified to include all mod AAs as well
restype_num = len([*restype_name_to_atom14_names.keys()])  # := 49
```

**29 Additional NCAAs**:
MSE, TPO, MLY, CME, PTR, SEP, SAH, CSO, PCA, KCX, CAS, CSD, MLZ, OCS, ALY, CSS, CSX, HIC, HYP, YCM, YOF, M3L, PFF, CGU, FTR, LLP, CAF, CMH, MHO

### 2. **Extended Atom Representation (14 → 25 atoms)**

#### **AlphaFold's Dense Representation**:
- Uses "atom14" representation (max 14 atoms per residue)
- Canonical amino acids fit within 14 atoms
- Example: ALA has 5 atoms (N, CA, C, CB, O)

#### **RareFold's Extended Representation**:
From `src/rarefold/common/residue_constants.py:1100`:
```python
# A compact atom encoding with 14 (now 25) columns
restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'CB', 'O', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    # ...
    # NCAAs with more atoms:
    'SAH': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'SD', "C5'", "C4'", "O4'", "C3'",
            "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'N6',
            'N1', 'C2', 'N3', 'C4'],  # 25 atoms!
}
```

**Why the extension**:
- Some NCAAs have complex modifications (e.g., SAH has an adenosine group)
- Need to represent up to **25 atoms** per residue
- All residues padded to same length for efficient batching

### 3. **NCAA-Specific Geometric Parameters**

Each NCAA has defined:

#### **a) Chi Angles** (Side-chain rotations)
Example from lines 65-93:
```python
chi_angles_atoms = {
    # Canonical
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
    # Noncanonical (Selenomethionine)
    'MSE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SE'], ['CB', 'CG', 'SE', 'CE']],
    # Note: SE (selenium) instead of SD (sulfur)
}
```

#### **b) Rigid Group Atom Positions** (3D coordinates relative to backbone)
Example from lines 435-461 (MSE vs MET):
```python
# Methionine (canonical)
'MET': [
    ['N', 0, (-0.521, 1.364, -0.000)],
    ['CA', 0, (0.000, 0.000, 0.000)],
    # ...
    ['SD', 5, (0.703, 1.695, 0.000)],  # Sulfur
    ['CE', 6, (0.320, 1.786, -0.000)],
],

# Selenomethionine (noncanonical)
'MSE': [
    ['N', 0, (-0.521, 1.364, -0.000)],
    ['CA', 0, (0.000, 0.000, 0.000)],
    # ...
    ['SE', 5, (0.703, 1.695, 0.000)],  # Selenium (different atom type!)
    ['CE', 6, (0.320, 1.786, -0.000)],
],
```

#### **c) Example: Phosphorylated Residues**

**SEP (Phosphoserine)** from lines 514-526:
```python
'SEP': [
    ['N', 0, (-0.529, 1.360, -0.000)],
    ['CA', 0, (0.000, 0.000, 0.000)],
    ['C', 0, (1.525, -0.000, -0.000)],
    ['CB', 0, (-0.518, -0.777, -1.211)],
    ['O', 3, (0.626, 1.062, -0.000)],
    ['OG', 4, (0.503, 1.325, 0.000)],      # Regular serine up to here
    ['P', 4, (1.937, 1.229, -0.846)],      # Phosphate group added!
    ['O1P', 4, (1.563, 0.735, -2.19)],     # Phosphate oxygens
    ['O2P', 4, (2.686, 2.647, -0.984)],
    ['O3P', 4, (2.921, 0.169, -0.141)],
],
```

### 4. **Van der Waals Radii Extensions**

From lines 953-962:
```python
van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
    'P': 1.8,      # For phosphorylated residues
    'A': 1.85,     # Arsenic (AS in CAS, CAF)
    'F': 1.47,     # Fluorine (in PFF, YOF, FTR)
    'H': 1.55,     # Mercury (HG in CMH)
}
```

---

## How Prediction Works: Step-by-Step with NCAAs

### **Input Processing**

1. **FASTA with NCAAs** (user provides):
   ```
   >1FU0_A
   MEKKEFHIVAETGIHARPATLLVQTASKFNSDINLEYKGKSVNLK-SEP-IMGVMSLGVGQGSDVTITVDGADEAEGMAAIVETLQKEGLA
   ```
   - Regular AAs: Single letter (M, E, K, ...)
   - NCAAs: Three-letter code in hyphens (-SEP-)

2. **MSA Generation** (HHblits):
   - NCAAs replaced with 'X' (unknown) for MSA search:
   ```
   >1FU0_A
   MEKKEFHIVAETGIHARPATLLVQTASKFNSDINLEYKGKSVNLKXIMGVMSLGVGQGSDVTITVDGADEAEGMAAIVETLQKEGLA
   ```
   - MSA provides evolutionary context from homologous proteins
   - AlphaFold can use MSA even though it doesn't know about NCAA

3. **Integer Sequence Mapping** (`get_int_seq()` in `predict_sc.py:133-164`):

   ```python
   # Example mapping:
   # Position 48: 'X' in MSA → NCAA 'SEP' in input

   all_AAs = ['ALA', 'ARG', ..., 'VAL',  # 0-19 (canonical)
              'UNK',                      # 20 (unknown)
              'MSE', 'TPO', ..., 'MHO']  # 21-49 (noncanonical)

   # Map sequence to integers
   int_protein_seq[48] = np.argwhere(all_AAs == 'SEP')[0][0]  # → 26
   ```

4. **Feature Generation** (`make_features()` in `predict_sc.py:77-126`):

   ```python
   # Create one-hot encoding (49-dimensional)
   feature_dict['aatype'] = np.eye(49)[int_protein_seq]

   # Position 48:
   # [0, 0, ..., 0, 1, 0, ..., 0]  # 1 at index 26 (SEP)
   #                ↑
   #           SEP position

   # Create atom mappings
   # For SEP (index 26):
   residx_atom14_to_atom37 = get_mapping_for_restype(26)
   # Returns: indices to map 25-atom representation to 37-atom representation

   # Atom37 representation:
   # [N, CA, C, O, CB, ..., P, O1P, O2P, O3P, ...]
   # 37 possible atom positions (superset of all possible atoms)

   # Atom14/25 representation (SEP specific):
   # [N, CA, C, CB, O, OG, P, O1P, O2P, O3P, '', '', ...]
   # 25 positions, padded with empty slots
   ```

### **Model Processing**

5. **Evoformer** (Same as AlphaFold):
   ```
   Input:
   - MSA representation: [N_sequences, N_residues, 256]
   - Pair representation: [N_residues, N_residues, 128]

   Processing:
   - MSA attention (within sequences)
   - Pair attention (between residue pairs)
   - Outer product mean (MSA → Pair)
   - Triangle multiplicative/additive updates

   Output:
   - Updated representations (learned evolutionary + structural patterns)
   ```

   **Key**: NCAA types are embedded in the one-hot encoding. The model learns:
   - "Position 48 has NCAA type 26 (SEP)"
   - "SEP is similar to SER but has extra phosphate group"
   - "Phosphate groups prefer specific local environments"

6. **Structure Module** (Same as AlphaFold):
   ```
   Input:
   - Pair representation from Evoformer
   - Single representation (per-residue)

   Processing:
   - IPA (Invariant Point Attention)
   - Generates rigid body transformations
   - Predicts backbone frames (N-CA-C)
   - Predicts side-chain torsions (chi angles)

   Output:
   - 3D coordinates for ALL atoms (up to 25 per residue)
   ```

   **NCAA-specific processing**:
   ```python
   # For position 48 (SEP):
   # 1. Predict backbone frame (same as canonical AAs)
   # 2. Predict chi angles (SEP has 1 chi angle, like SER)
   # 3. Place atoms using rigid_group_atom_positions['SEP']

   # SEP atoms:
   final_atoms = [
       N:   (backbone frame),
       CA:  (backbone frame),
       C:   (backbone frame),
       CB:  (chi1 rotation from CA),
       O:   (backbone frame),
       OG:  (chi1 rotation from CB),
       P:   (chi1 rotation from OG),    # Phosphate group!
       O1P: (fixed relative to P),
       O2P: (fixed relative to P),
       O3P: (fixed relative to P),
   ]
   ```

### **Output Generation**

7. **Atom Coordinate Prediction**:
   ```
   For each residue:
   - Use atom14/25 representation internally
   - Convert to atom37 representation for output
   - Apply masks (some atoms don't exist for certain residues)

   Example for SEP:
   atom37_positions[48] = [
       N:   (x1, y1, z1),
       CA:  (x2, y2, z2),
       ...
       P:   (x10, y10, z10),  # Phosphate position predicted!
       ...
   ]
   atom37_mask[48] = [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 0, 0, ...]
                      ↑  ↑              ↑           ↑
                      N  CA             P          unused atoms
   ```

8. **PDB File Generation** (`save_structure()` in `predict_sc.py:233-255`):
   ```python
   # Convert predictions to PDB format
   # NCAAs saved to chain B for easier visualization

   # Example PDB output:
   ATOM   1048  N   SEP B  48      12.345  23.456  34.567  1.00 85.32
   ATOM   1049  CA  SEP B  48      13.456  24.567  35.678  1.00 87.21
   ATOM   1050  C   SEP B  48      14.567  25.678  36.789  1.00 86.54
   ATOM   1051  O   SEP B  48      15.678  26.789  37.890  1.00 84.32
   ATOM   1052  CB  SEP B  48      16.789  27.890  38.901  1.00 88.12
   ATOM   1053  OG  SEP B  48      17.890  28.901  39.012  1.00 89.45
   ATOM   1054  P   SEP B  48      18.901  29.012  40.123  1.00 82.11  # Phosphate!
   ATOM   1055  O1P SEP B  48      19.012  30.123  41.234  1.00 81.23
   ATOM   1056  O2P SEP B  48      20.123  31.234  42.345  1.00 80.45
   ATOM   1057  O3P SEP B  48      21.234  32.345  43.456  1.00 79.87
   ```

---

## Training: How RareFold Learned to Handle NCAAs

### **Training Data**
From the paper (bioRxiv 2025.05.19.654846):
- Structures from PDB containing noncanonical amino acids
- MSAs generated for each structure
- Ground truth: experimental structures with NCAA coordinates

### **Fine-tuning Process**

1. **Start with AlphaFold weights**:
   - Pre-trained on canonical amino acids
   - Understands general protein structure principles
   - Knows backbone geometry, secondary structure, etc.

2. **Extend the model**:
   ```python
   # Original AlphaFold embedding
   aatype_embedding = Linear(20 → 256)  # 20 AA types

   # RareFold embedding
   aatype_embedding = Linear(49 → 256)  # 49 AA types

   # New weights for NCAAs initialized randomly or from similar canonical AAs
   # Example: SEP weights initialized from SER weights
   ```

3. **Fine-tune on NCAA structures**:
   - Model learns:
     - NCAA-specific geometric constraints
     - Which environments favor certain NCAAs
     - How phosphate/methyl/other groups affect structure

4. **Parameter files**:
   - `params20000.npy`: For prediction (20K training steps)
   - `finetuned_params25000.npy`: For design (25K steps, optimized for binder design)

---

## Design: How It Differs from Prediction

### **Prediction** (AlphaFold-like):
```
Given: Amino acid sequence (with NCAAs)
Predict: 3D structure
```

### **Design** (Inverse problem):
```
Given: Target protein structure
Design: Binder sequence (optimizing NCAA usage)
```

### **Design Process** (`mc_design_length_var_batch.py`):

1. **Start with random peptide sequence**:
   ```python
   # Initialize with random AAs (including NCAAs)
   binder_seq = random_sample(['ALA', 'ARG', ..., 'MSE', 'MLY', ...])
   # Example: ['MET', 'SEP', 'ARG', 'MLY', 'PHE', ...]
   ```

2. **Create combined sequence**:
   ```python
   # Concatenate target + binder
   full_seq = target_seq + binder_seq
   # Example: [target_protein (200 residues)] + [binder (12 residues)]
   ```

3. **Predict structure** (using RareFold):
   ```
   structure = RareFold.predict(full_seq)
   # Returns 3D coordinates for all atoms
   ```

4. **Calculate loss**:
   ```python
   # Measure quality of binding
   loss = (
       interface_distance * (1/plddt) +  # Want close contact, high confidence
       inter_clash_fraction +             # Minimize inter-molecular clashes
       intra_clash_fraction               # Minimize intra-molecular clashes
   )
   ```

5. **Mutate and iterate**:
   ```python
   # Monte Carlo optimization
   for iteration in range(1000):
       # Randomly mutate one position
       mutated_seq = mutate(binder_seq, position=random, aa=random)

       # Predict new structure
       new_structure = RareFold.predict(target + mutated_seq)

       # Calculate new loss
       new_loss = calculate_loss(new_structure)

       # Accept if better
       if new_loss < best_loss:
           binder_seq = mutated_seq
           best_loss = new_loss
   ```

### **Why NCAAs Help in Design**:

1. **Expanded chemical space**:
   - Canonical AAs: 20 options per position
   - With NCAAs: 49 options per position
   - More chances to find optimal binding

2. **Novel interactions**:
   - Phosphorylated AAs: Strong electrostatic interactions
   - Methylated lysines: Altered pKa, different H-bonding
   - Modified cysteines: Different disulfide chemistry

3. **Experimental validation** (from paper):
   - EvoBindRare successfully designed cyclic binders with NCAAs
   - High-affinity binding achieved
   - NCAAs provided unique binding modes not possible with canonical AAs

---

## Key Differences: RareFold vs AlphaFold

| Aspect | AlphaFold | RareFold |
|--------|-----------|----------|
| **AA Vocabulary** | 20 canonical | 49 (20 canonical + 29 noncanonical) |
| **Atom Representation** | 14 atoms max | 25 atoms max |
| **Residue Constants** | Standard | Extended with NCAA parameters |
| **Training Data** | PDB (canonical) | PDB + NCAA structures |
| **Use Cases** | Structure prediction | Structure prediction + binder design with NCAAs |
| **Model Architecture** | Evoformer + Structure Module | **Same** (Evoformer + Structure Module) |
| **Recycling** | Yes | **Same** |
| **MSA Processing** | Yes | **Same** (NCAAs as 'X' in MSA) |
| **Output** | Atom coordinates | **Same** + NCAA atom coordinates |

---

## Summary

**RareFold = AlphaFold + NCAA Support**

### What Stayed the Same:
✓ Core neural network architecture (Evoformer)
✓ Structure module (IPA, rigid body transformations)
✓ Recycling mechanism
✓ MSA processing
✓ Training methodology

### What Changed:
✗ Extended amino acid vocabulary (20 → 49)
✗ Extended atom representation (14 → 25)
✗ Added NCAA-specific geometric parameters
✗ Fine-tuned on NCAA-containing structures
✗ Added design capabilities (EvoBindRare)

### How It Works:
1. **MSA**: NCAAs replaced with 'X' (AlphaFold can still use evolutionary info)
2. **Embedding**: NCAAs get their own one-hot encoding (positions 21-49)
3. **Processing**: Same neural network learns NCAA-specific patterns
4. **Structure**: Uses NCAA-specific atom definitions to place atoms correctly
5. **Output**: Predicts 3D coordinates for all atoms, including NCAA modifications

### Why It's Powerful:
- Leverages AlphaFold's proven architecture
- Extends chemical alphabet without reinventing the wheel
- Enables design of novel binders with expanded chemistry
- Experimentally validated for both prediction and design

---

## References

1. **AlphaFold 2**: Jumper et al. (2021) "Highly accurate protein structure prediction with AlphaFold" Nature
2. **RareFold**: Li Q, Daumiller D, Zuo F, et al. (2025) "RareFold: Structure prediction and design of proteins with noncanonical amino acids" bioRxiv
3. **Code**: `src/rarefold/` package - extended from AlphaFold codebase
4. **Parameters**: Fine-tuned from AlphaFold weights (evident from parameter renaming in `predict_sc.py`)
