# Repository Overview

This repository includes the logical error rates, Jupyter notebooks, and Python files used in the research article [https://arxiv.org/abs/2412.07841](https://arxiv.org/abs/2412.07841). This work can be cited using the DOI: [10.5281/zenodo.14639657](https://zenodo.org/records/14865271).
The Detector Error Models have not been uploaded in the github but can be downloaded from the [Zenodo repository](https://zenodo.org/records/14865271).

## Directory Structure

### `data/logical_error/`:

Contains all the files storing the logical error rates computed from the simulations

### Python Files

Contains the class and functions needed for the simulations:

- Python files used to **construct the surface code**:
  - **`surface_code_no_detection_loss_circuit.py`** (without LDU)
  - **`surface_code_standard_LDU.py`** (with standard LDU)
  - **`surface_code_teleportation_LDU.py`** (with teleportation LDU)



- Python files used to **sample losses**:
  - **`sampler_standard_LDU.py`** (with standard LDU)
  - **`sampler_teleportation_LDU.py`** (with teleportation LDU)
  - **`sampler_teleportation_LDU_Z_bias.py`** (with teleportation LDU and Z error on the remaining atom)



- Python files used to **build the DEM**:
  - **`dem_builder_standard_LDU.py`** (with standard LDU)
  - **`dem_builder_teleportation_LDU.py`** (with teleportation LDU)
  - **`dem_builder_teleportation_LDU_Z_bias.py`** (with teleportation LDU and Z error on the remaining atom)



### Jupyter notebooks

Contains the code to build the DEM, compute the logical error rate and plot the results for the various cases:

- Notebooks used to **generate the DEM**:
  - **`build_dem_standard_LDU.ipynb`** (circuit in the Z-basis)
  - **`build_dem_standard_LDU_X_basis.ipynb`**  (circuit in the X-basis)
  - **`build_dem_teleportation_LDU.ipynb`** 
  - **`build_dem_teleportation_LDU_X_basis.ipynb`**
  - **`build_naive_dem_standard_LDU.ipynb`**
  - **`build_naive_dem_teleportation_LDU.ipynb`**
  - **`build_dem_teleportation_LDU_Z_bias.ipynb`**

 
- Notebooks used to **simulate the surface code and compute the logical error rates**:
  - **`simulation surface code without LDU.ipynb`**
  - **`simulation_standard_LDU.ipynb`**
  - **`simulation_standard_LDU_X_basis.ipynb`**
  - **`simulation_teleportation_LDU.ipynb`**
  - **`simulation_teleportation_LDU_X_basis.ipynb`**
  - **`simulation_naive_standard_LDU.ipynb`** 
  - **`simulation_naive_teleportation_LDU.ipynb`**
  - **`simulation_teleportation_LDU_Z_bias.ipynb`**


- Notebooks used to **plot the results**:
  - **`plot_standard_LDU.ipynb`** 
  - **`plot_standard_LDU_X_basis.ipynb`**
  - **`plot_teleportation_LDU.ipynb`**
  - **`plot_teleportation_LDU_X_basis.ipynb`**
  - **`plot_naive_standard_LDU.ipynb`**
  - **`plot_naive_teleportation_LDU.ipynb`**
  - **`plot_teleportation_LDU_Z_bias.ipynb`**


## Dependencies

The project has been run using the following packages (with specific versions):

- Python 3.9.5
- stim 1.12.1
- Pymatching 2.2.0
- NumPy 1.26.4
- matplotlib 3.8.3
- scipy 1.12.0
