#!/bin/bash
#SBATCH -A LRN037
#SBATCH -J matey
#SBATCH -o %x-%j.out
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 5
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
source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"

export MIOPEN_USER_DB_PATH=/mnt/bb/$USER/MIOPEN$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442

export run_name="demo_avit_test"
export yaml_config=./config/Demo_MW_avit.yaml
export NN=1
srun -N$NN -n$((NN*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp > log_avit 2>&1 &

export run_name="demo_svit_test"
export yaml_config=./config/Demo_MW_svit.yaml
srun -N$NN -n$((NN*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp > log_svit 2>&1 &

export run_name="demo_vit_test"
export yaml_config=./config/Demo_MW_vit.yaml
srun -N$NN -n$((NN*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp > log_vit 2>&1 &


export run_name="demo_turt_test"
export yaml_config=./config/Demo_JHUTDB_TT.yaml
export NN=2
srun -N$NN -n$((NN*8)) -c7 --gpu-bind=closest python basic_usage.py \
--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp > log_turt 2>&1 &

wait

