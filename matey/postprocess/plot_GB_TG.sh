#!/bin/bash
#SBATCH -A LRN037
#SBATCH -J matey
#SBATCH -o %x-%j.out
#SBATCH -t 00:45:00
##SBATCH -p extended
#SBATCH -p batch
#SBATCH -N 1
##SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME

source /lustre/orion/world-shared/stf218/junqi/forge/matey-env.sh

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

python plots_GB_fields_TG.py

srun -N1 -n$((NN*8)) -c7 --gpu-bind=closest python inference_GB_TaylorGreen.py --plotKE --use_ddp 


