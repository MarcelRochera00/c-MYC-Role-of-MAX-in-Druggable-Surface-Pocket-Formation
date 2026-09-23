# MYC-MAX MD Analysis Pipeline

This repository contains the Python scripts used for analyzing Molecular Dynamics simulations of the MYC-MAX complex and isolated c-MYC. The analysis pipeline directly mirrors the methodology and results discussed in the associated paper: *MYC-MAX project for ACS Journal Submissions*.

## Results Directory Structure
All scripts automatically output to the `Results/` directory:
- `Results/Individual/`: RMSD, RMSF, and Rg for single-protein replicas (isolated c-MYC).
- `Results/Comparison/`: Comparative analysis between isolated c-MYC and c-MYC in the complex.
- `Results/Complex/`: Interface Contacts and Salt Bridges for the MYC:MAX dimer.
- `Results/Pockets/`: MDpocket descriptors, sub-pocket clustering, and multi-method cross-validation tables.

---

## Workflow & Methodology

The files have been intuitively renamed and numbered to follow the analytical steps of the paper. To ensure all dependencies are met, run the scripts in the following order:

### 1. `01_isolated_myc_analysis.py`
- **Paper Section:** 1. Conformational disorder of isolated c-MYC
- **Purpose:** Primary analysis of isolated c-MYC replicas. It calculates the backbone RMSD, per-residue RMSF, and Radius of Gyration (Rg) to demonstrate the high structural flexibility and lack of stable 3D folded structure of the isolated c-MYC.
- **Outputs:** Saves metrics and plots to `Results/Individual/`.

### 2. `02_comparative_myc_analysis.py`
- **Paper Section:** 2. MAX binding stabilises c-MYC conformation
- **Purpose:** Compares the structural dynamics (RMSD, RMSF, and Rg) of c-MYC in isolation versus c-MYC in the MAX complex. This step demonstrates how MAX binding drastically reduces c-MYC's conformational drift and stabilizes its structure.
- **Outputs:** Saves comparative plots to `Results/Comparison/`.

### 3. `03_myc_max_complex_analysis.py`
- **Paper Section:** 3. The MYC-MAX binding interface is stable and persistent
- **Purpose:** Analyzes the MYC-MAX complex to confirm binding interface stability. It monitors inter-chain atomic contacts, calculates the MM-GBSA per-residue energy decomposition and persistent inter-chain salt bridges over the trajectory.
- **Outputs:** Saves Interface Contacts, MM-GBSA hotspots and Salt Bridges results to `Results/Complex/`.

### 4. `04_pocket_identification.py`
- **Paper Section:** 4.1 Trajectory-based pocket detection
- **Purpose:** Uses MDpocket frequency grid clustering on the pooled complex trajectory to identify contiguous sub-pockets at the MYC-MAX interface. Extracts the core lining residues.
- **Outputs:** Saves sub-pocket spatial properties to `Results/Pockets/`.

### 5. `05_pocket_characterization.py`
- **Paper Section:** 4.1 Trajectory-based pocket detection (continued)
- **Purpose:** Evaluates physicochemical properties for the identified pockets over time. It calculates Mean Pocket Volume, Hydrophobicity Score, Mean Local Hydrophobic Density, and Polarity Score.
- **Outputs:** Saves summary tables and plots to `Results/Pockets/`.

### 6. `06_multi_method_validation.py`
- **Paper Section:** 4.3 Triple-method validation
- **Purpose:** Integrates the results from three independent approaches: MDpocket cavity detection, MM-GBSA per-residue energy decomposition, and persistent salt bridge analysis. Identifies the highest-confidence interface hotspots (residues scoring at least 2 out of 3 methods, e.g., in sub-pocket SP2).
- **Outputs:** Generates the final consensus table and cross-validation summaries in `Results/Pockets/`.

### Extra Scripts
- `individual_residue_testing.py`: Helper script for local testing of specific residue properties.

---

## 🛠 Prerequisites
- **Python 3.8+**
- **Libraries:** `MDAnalysis`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `gridData`
- **External Tools:** `gmx_MMPBSA` (for free energy), `mdpocket` (for cavity analysis).

## Data Handling & GitHub
Due to the significant size of Molecular Dynamics trajectories (individual `.xtc` files range from 100MB to 7GB; total project >50GB), raw data files are **excluded** from this repository. 

**To use this repository with your own data:**
1. Maintain the directory structure.
2. Ensure your trajectories are named according to the definitions in the `CONFIG` section of the scripts.
3. The scripts use relative path routing (via `PROJECT_ROOT`), so they will work immediately as long as the internal folder hierarchy is preserved.
