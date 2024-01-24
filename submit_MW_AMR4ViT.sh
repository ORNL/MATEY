#!/bin/bash
#SBATCH -A LRN037
#SBATCH -J job_MW
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 16
##SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export run_name="demo"
export config="basic_config"   # options are "basic_config" for all or swe_only/comp_only/incomp_only/swe_and_incomp
export yaml_config=./config.yaml

module load rocm/6.0.0
source /lustre/orion/proj-shared/lrn037/gounley1/conda600whl/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/lrn037/gounley1/conda600whl
module load cray-parallel-netcdf

export MIOPEN_USER_DB_PATH=/mnt/bb/zhangp/MIOPEN$SLURM_JOB_ID
mkdir $MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
