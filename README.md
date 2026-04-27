# Recursive Least Squares Identification & Fisher Information Analysis

This repository contains a Python implementation for identifying a discrete-time 3x3 nonlinear system using the Recursive Least Squares (RLS) algorithm. 

A primary focus of this project is the empirical validation of Persistent Excitation (PE) and its direct mathematical relationship with parameter convergence. The script tracks the minimum eigenvalue of the Fisher Information Matrix (FIM) as a definitive measure of identifiability.

## Features

An object-oriented architecture segregates the system dynamics from the estimator. Running the script generates a comprehensive four-panel visualization:

1. **3D State Space Evolution:** Validates bounded system trajectories during identification.
2. **Parameter Convergence:** Tracks estimates against the true system values.
3. **FIM Analysis:** Demonstrates the unbounded growth of the FIM minimum eigenvalue under persistently exciting inputs.
4. **Parameter Uncertainty Decay:** Visualizes the shrinking parameter covariance trace on a logarithmic scale.

## Dependencies

* `numpy`
* `matplotlib`

## Usage

Clone the repository and execute the main script:

git clone https://github.com/szk6437/Nonlinear-System-Identification.git
cd Nonlinear-System-Identification
python NonlinearDataDrivenControl.py

## License
MIT License
