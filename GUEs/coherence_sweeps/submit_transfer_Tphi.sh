#!/bin/bash

###### modify params here
PARAM_KEY="Gamma_phi_transfer"
NUM_PTS=9
BASE=2.15
PREFACTOR=100
NTH=0.0
NUM_CPUS=16
SBATCH_SCRIPT="_submit_transfer_Tphi.sh"
######

cat > $SBATCH_SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=coherence_sweep_cav_T1
#SBATCH -o out/output-$PARAM_KEY-%a.txt -e out/errors-$PARAM_KEY-%a.txt
#SBATCH --array=0-$NUM_PTS
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$NUM_CPUS
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

ARRAY_IDX_LIST=($(seq 0 1 $NUM_PTS))

module load miniconda
conda activate qram_fidelity
python run_GUE_coherence_sweep.py --param_key=$PARAM_KEY --idx=\${ARRAY_IDX_LIST[\${SLURM_ARRAY_TASK_ID}]} --num_pts=$NUM_PTS --base=$BASE --prefactor=$PREFACTOR --nth=$NTH --num_cpus=$NUM_CPUS
EOL

chmod +x $SBATCH_SCRIPT
sbatch $SBATCH_SCRIPT
