#!/bin/bash -l
#SBATCH -A ccsd
#SBATCH -t 12:00:00
#SBATCH -p gpu_p100
##SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
##SBATCH -G 2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH -J demo
#SBATCH --mem=0g

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export run_name="demo"
export config="basic_config"   
export yaml_config=./config/mpp_avit_ti_config.yaml

#source /home/6pz/virtual/mpp/bin/activate 
conda activate mpp2

export MIOPEN_USER_DB_PATH=$PWD/MIOPEN$SLURM_JOB_ID
mkdir $MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=2 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:29500 \
train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp

