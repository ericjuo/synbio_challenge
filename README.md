# 2025 SynBio Challenge  
This repository demonstrates the in silico directed evolution of avGFP variants using EvolvePro and coarse-grained molecular dynamics simulations with GROMACS.  
## Install

### EvolvePro
Clone the EvolvePro repository and set up the Conda environment:Download EvolvPro from source:  
```
git clone https://github.com/mat10d/EvolvePro.git
cd EvolvePro
conda env create -f environment.yml
conda activate evolvepro
```

### Gromacs

Download and compile GROMACS from source:
```
tar xfz gromacs-2024.5
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DGMX_GPU=CUDA
make
make check
sudo make install
source /usr/local/gromacs/bin/GMXRC

```
 Ensure you have CUDA installed and properly configured for GPU acceleration.  

### Martinize
Install vermouth and martinize2 for generating coarse-grained topologies:
```
pip install vermouth
```

## Usage 
We predict avGFP variants using the EvolvePro program. This tool performs in silico directed evolution guided by a custom-defined fitness function.  

### Fitness function
The fitness combines:  
- Predicted brightness (from a trained brightness prediction model)
- ΔRMSF: The change in root mean square fluctuation between 300K and 400K simulated environments (using coarse-grained models)  

## Main Script
The evolution workflow is implemented in the Jupyter notebook:  
```
avGFP_evolution.ipynb
```
The notebook includes steps for candidate generation, fitness prediction, and top N variant selection.  

## Directory Structure
### `data/evoltuion/results` – Evolution Rounds 
This directory contains:

- Initial variant selections

- Top N mutations proposed by EvolvePro in each generation

- Evaluation results of predicted variants (e.g., fitness scores, brightness, ΔRMSF)Initial selections and top N mutations suggested by EvolvePro are in data/evolution/results. Evaluation results are in this directory as well.  

### `md/rawdata/` – Coarse-Grained Simulations
This directory contains trajectory data:  
-   `simulation at 300K/`- Simulations of variants at 300K  
-   `simulation at 400K/` - Simulatinos of variants at 400K  

These trajectories are used to compute ΔRMSF for each variant as part of the fitness evaluation.  



