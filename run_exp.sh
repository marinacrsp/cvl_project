#!/bin/bash
#SBATCH  --output=logs/%j.out
#SBATCH  --cpus-per-task=4
#SBATCH  --gres=gpu:1
#SBATCH  --constraint='titan_xp'
#SBATCH  --mem=20G
nvidia-smi

source /scratch_net/ken/mcrespo/conda/etc/profile.d/conda.sh # TODO: SET.
conda activate pytcu11


# # NOTE: Uncomment when running multi-gpu script.
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_PORT=$MASTER_PORT"

# # module load readline
# # master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# # master_addr=localhost
# # export MASTER_ADDR=$master_addr
# export MASTER_ADDR=$(hostname -s)

# echo "MASTER_ADDR: $MASTER_ADDR"

# python -u multi_gpu/main.py
# python -u multi_vol/main.py
python -u single_vol/main.py
# python -u single_vol_multigpu/main.py
