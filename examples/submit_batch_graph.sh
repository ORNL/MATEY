#!/bin/bash
#SBATCH -A LRN037
#SBATCH -J matey
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export config="basic_config" 
#export run_name="demo_graph_svit"
#export yaml_config=./config/Demo_graph_svit.yaml
#export run_name="demo_graph_turbt"
#export yaml_config=./config/Demo_graph_turbt.yaml

export run_name="demo_graph_vit"
export yaml_config=./config/Demo_graph_vit.yaml

##conda env with rocm 6.0.0
#module load rocm/6.0.0
#source /lustre/orion/proj-shared/lrn037/gounley1/conda600whl/etc/profile.d/conda.sh
#conda activate /lustre/orion/proj-shared/lrn037/gounley1/conda600whl
#module load cray-parallel-netcdf/1.12.3.9

##conda env with rocm 6.0.0 in world-shared
#source /lustre/orion/world-shared/lrn037/gounley1/env600.sh
source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442

export TF_FORCE_GPU_ALLOW_GROWTH=true

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
