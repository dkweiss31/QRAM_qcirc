#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=ramsey_coherent_one_cav
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-200
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

###### modify params here
EPS=0.01
CAV_DIM=9
NUM_CAVS=2
NSTEPS=200000
######

ARRAY_IDX_LIST=($(seq 0 1 $((SLURM_ARRAY_TASK_COUNT-1))))

module load miniconda
conda activate qram_fidelity
python run_Ramsey_coherent.py \
--idx="${ARRAY_IDX_LIST[${SLURM_ARRAY_TASK_ID}]}" \
 --num_pts="$SLURM_ARRAY_TASK_COUNT" \
 --eps=$EPS \
 --cav_dim=$CAV_DIM \
 --num_cavs=$NUM_CAVS \
 --nsteps=$NSTEPS
