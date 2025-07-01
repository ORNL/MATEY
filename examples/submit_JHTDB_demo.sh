#!/bin/bash
#SBATCH -A lrn037
#SBATCH -J matey
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
##SBATCH -p extended
#SBATCH -N 2
##SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export run_name="demo"
export config="basic_config" 
export yaml_config=./config/Demo_JHUTDB_TT.yaml

source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
#"/tmp/cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
##export NCCL_DEBUG=INFO 

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
