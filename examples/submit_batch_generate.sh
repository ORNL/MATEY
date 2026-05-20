#!/bin/bash
#SBATCH -A LRN037
#SBATCH -J matey
#SBATCH -o %x-%j.out
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1
##SBATCH -q debug
#SBATCH -C nvme

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export config="basic_config" 
# export model_dir="./Demo_Diffusion_CIFAR10_3level_Ti_1e-3lr/basic_config/demo_diffusion/"
# export model_dir="./Demo_Diffusion_CIFAR10_UNet_1e-3lr/basic_config/demo_diffusion/"
# export model_dir="./Demo_Diffusion_CIFAR10_3level_Ti_newembskip_1e-3lr/basic_config/demo_diffusion/"

# export model_dir="./Demo_Diffusion_MW_cond_S_newembskip_lt5/basic_config/demo_diffusion/"
# export model_dir="./Demo_Diffusion_MW_cond_S_newembskip_1e-3lr/basic_config/demo_diffusion/"

# export model_dir="./Demo_Diffusion_MW_UNet_1e-3lr/basic_config/demo_diffusion/"
# export model_dir="./Demo_Diffusion_MW_cond_UNet_1e-3lr/basic_config/demo_diffusion/"
# export model_dir="./Demo_Diffusion_MW_cond_UNet_newemb_lt5/basic_config/demo_diffusion/"

export model_dir="./Demo_Diffusion_MW_cond_TurbT_3level_S_lt5/basic_config/demo_diffusion/"


# export output_dir="./CIFAR10_generation_outputs/"
# export output_dir="./MW_generation_outputs/"
# export output_dir="./MW_cond_generation_outputs_batches/"
# export output_dir="./MW_cond_generation_outputs_batches_lt5/turbt/"
# export output_dir="./MW_cond_generation_outputs_batches_lt5/UNet/"
export output_dir="./MW_cond_generation_outputs_batches_lt5/TurbTMod/"

##conda env with rocm 6.0.0
#module load rocm/6.0.0
#source /lustre/orion/proj-shared/lrn037/gounley1/conda60s0whl/etc/profile.d/conda.sh
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

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*1)) -c7 --gpu-bind=closest python basic_generate.py \
--config $config --model_dir $model_dir --output_dir $output_dir --use_ddp 
