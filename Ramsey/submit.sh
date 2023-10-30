#!/bin/bash

###### modify params here
NUM_PTS=101
NUM_ARRAY=$NUM_PTS
NUM_IDXS=$((NUM_ARRAY-1))
NUM_CPUS=1
EPS=0.01
CAV_DIM=9
NUM_CAVS=2
NSTEPS=200000
######

SBATCH_SCRIPT="tmp_slurm_script.sh"

cat > $SBATCH_SCRIPT <<EOL
#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=ramsey_coherent_one_cav
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-$NUM_IDXS
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$NUM_CPUS
#SBATCH --mem-per-cpu=10G
#SBATCH --time=04:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

ARRAY_IDX_LIST=($(seq 0 1 $NUM_IDXS))

module load miniconda
conda activate qram_fidelity
python run_Ramsey_coherent.py --idx=\${ARRAY_IDX_LIST[\${SLURM_ARRAY_TASK_ID}]} --num_pts=$NUM_PTS --eps=$EPS --cav_dim=$CAV_DIM --num_cavs=$NUM_CAVS --nsteps=$NSTEPS
EOL

chmod +x $SBATCH_SCRIPT
sbatch $SBATCH_SCRIPT
