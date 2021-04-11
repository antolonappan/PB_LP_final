#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH -J PB_largepatch
#SBATCH -o pb_pe.out
#SBATCH -e pb_pe.err
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it



source /global/homes/l/lonappan/.bashrc
conda activate PB_LP

module load cray-hdf5-parallel
export HDF5_USE_FILE_LOCKING=FALSE

cd /global/u2/l/lonappan/workspace/PB_LP_analysis/PB_LP

mpirun -np 96 python interface.py