#!/bin/bash
#SBATCH -A LRN037
#SBATCH -J matey-pretB
#SBATCH -o %x-%j.out
##SBATCH -t 12:00:00
#SBATCH -t 01:40:00
#SBATCH -p batch
##SBATCH -p extended
#SBATCH -N 16
#SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export run_name="demoB_debug"
export config="basic_config" 
export yaml_config=./config/Pretraining_TT_B.yaml

source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
#"/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
#export NCCL_DEBUG=INFO 
#export NCCL_ASYNC_ERROR_HANDLING=1

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
