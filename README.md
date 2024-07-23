# Low-Synchronization GMRES Algorithms

This repository contains the implementation of three GMRES (Generalized Minimal Residual) algorithms with a focus on low synchronization, tested on symmetric matrices from the SuiteSparse Matrix Collection.

## Overview

The implemented GMRES algorithms are:
1. **IGS-GMRES with 2 Synchronizations**: Uses two Gauss-Seidel iterations to maintain orthogonality.
2. **Low-Synchronization MGS-GMRES with 1 Synchronization**: Uses the modified Gram-Schmidt process with one synchronization.
3. **Original MGS-GMRES by Saad and Schultz**: Implements the original MGS-GMRES algorithm as described by Saad and Schultz.

## Setup

To set up the project and run the implementations, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/low-sync-gmres.git
    cd low-sync-gmres
    ```

2. **Install the required Python packages:**
    ```bash
    pip install numpy scipy
    ```

3. **Download the required symmetric matrix:**
    - Ensure you have the `bcsstk26.mtx` file from this respository or the SuiteSparse Matrix Collection and place it in the same directory as the scripts.

## Running the Scripts

The repository contains three scripts for the GMRES implementations and a comparison script:

1. **IGS-GMRES with 2 Synchronizations:** `igs_gmres.py`
2. **Low-Synchronization MGS-GMRES with 1 Synchronization:** `ls_mgs_gmres.py`
3. **Original MGS-GMRES by Saad and Schultz:** `original_mgs_gmres.py`

To compare the performance of the three implementations, run the comparison script:
```bash
python compare_gmres.py