#!/bin/bash

#SBATCH --account=ec158

# Runtime and memory
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00

#SBATCH --ntasks-per-node=1 # number of cores, max 128 on fox

singularity exec -H /fp/projects01/ec158/abl_scm_perturbation_study/ docker/abl_scm_venv.sif python3 -u main.py
