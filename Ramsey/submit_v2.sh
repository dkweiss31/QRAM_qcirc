#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=ramsey_coherent_one_cav
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-200
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

###### modify params here
exp_type="ramsey"
omega_d_vals="(3.2,3.6)"
eps=0.01
cav_dim=7
num_cavs=2
delay_times="(0,2000,301)"
nsteps=200000
temp=0.1
destructive_interference=0
include_stark_shifts=0
interference_scale=1
thermal_time=1000.0
######

ARRAY_IDX_LIST=($(seq 0 1 $((SLURM_ARRAY_TASK_COUNT-1))))

module load miniconda
conda activate qram_fidelity
python run_Ramsey_coherent_v2.py \
--idx="${ARRAY_IDX_LIST[${SLURM_ARRAY_TASK_ID}]}" \
 --num_pts="$SLURM_ARRAY_TASK_COUNT" \
 --exp_type=$exp_type \
 --omega_d_vals=$omega_d_vals \
 --eps=$eps \
 --cav_dim=$cav_dim \
 --num_cavs=$num_cavs \
 --delay_times=$delay_times \
 --nsteps=$nsteps \
 --temp=$temp \
 --destructive_interference=$destructive_interference \
 --interference_scale=$interference_scale \
 --include_stark_shifts=$include_stark_shifts \
 --thermal_time=$thermal_time
