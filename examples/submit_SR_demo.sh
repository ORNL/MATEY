#!/bin/bash
#SBATCH -A m4724
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J matey
#SBATCH -o %x-%j.out
#SBATCH -t 00:01:00
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 32

export OMP_NUM_THREADS=1

export master_node=$SLURMD_NODENAME
export config="basic_config" 
export run_name="demo_SR"
export yaml_config=./config/BLASNET_ti_baseline_SR.yaml

export SLURM_CPU_BIND="cores"

module load conda
module load cuda
conda activate matey_env

export PYTHONPATH="${PYTHONPATH}:$(dirname "$PWD")"


export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
# srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpu-bind=closest python basic_usage.py \
# --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp

srun -N1 -n2 -c7 bash -c '
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
python basic_usage.py --run_name SR_TT_test --config basic_config \
  --yaml_config ./config/BLASNET_ti_baseline_SR.yaml --use_ddp'