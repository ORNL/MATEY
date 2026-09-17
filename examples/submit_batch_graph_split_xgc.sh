#!/bin/bash
#SBATCH -A fus183
#SBATCH -J matey
#SBATCH -o %x-%j-xgcsplit.out
##SBATCH -t 08:00:00
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -p extended
#SBATCH -N 18
#SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export config="basic_config" 

##conda env with rocm 6.0.0
#module load rocm/6.0.0
#source /lustre/orion/proj-shared/lrn037/gounley1/conda600whl/etc/profile.d/conda.sh
#conda activate /lustre/orion/proj-shared/lrn037/gounley1/conda600whl
#module load cray-parallel-netcdf/1.12.3.9

##conda env with rocm 6.0.0 in world-shared
#source /lustre/orion/world-shared/lrn037/gounley1/env600.sh
source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
#source /lustre/orion/lrn037/proj-shared/zhangp/matey-env/load-matey-rocm720.sh
export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442

export TF_FORCE_GPU_ALLOW_GROWTH=true

#srun -n 200 python ../matey/data_utils/graph_datasets.py --dataset "xgc" --nooverwrite

export NN=2

export run_name="demo_graph_vit_lt1"
export yaml_config=./config/Fusion_Seed_AR_xgc.yaml

srun -N$NN -n$((NN*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp > "xgc_matey_log_vitbaseline_$SLURM_JOB_ID" &


export NN=16
export run_name="demo_graph_vit_split_lt1"
export yaml_config=./config/Fusion_Seed_AR_xgcsplit.yaml
srun -N$NN -n$((NN*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp > "xgc_matey_log_vitsplit_$SLURM_JOB_ID" 2>&1 &

wait
