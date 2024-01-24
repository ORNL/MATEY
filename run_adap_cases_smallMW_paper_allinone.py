import os

base_dir = "AMR4ViT_SmallMW_posbias_lr1e-4_adap_MLP"
setlr=True

model_attention=["avit", "svit", "vit"]
model_size=["ti","s","b"]
adaptivity=[True, False]

model_params = {
    "ti": {"embed_dim": 192, "num_heads": 3},
    "s": {"embed_dim": 384, "num_heads": 6},
    "b": {"embed_dim": 768, "num_heads": 12},
}
model_att_params = {
    "avit": {"time_type": 'attention', "space_type": 'axial_attention'},
    "svit": {"time_type": 'all2all_time', "space_type": 'all2all'},
    "vit_all2all": None,
}

config_src ="MW_baseline_small.yaml"
slurm_src = "submit_MW_AMR4ViT.sh"

with open(slurm_src, "r") as file:
    template_lines = file.readlines()
slurm_script_path = os.path.join(base_dir, "submit.sh")   

njobs=len(model_attention)*len(model_size) + 3 #3 for svit ad ti, s, b

slurm_script_content=[]
for line in template_lines:
    if "job_MW" in line:
        slurm_script_content.append(line.replace("job_MW","job_SmallMW"))
    elif "#SBATCH -N 16" in line:
        totalnodes=4*njobs
        slurm_script_content.append(line.replace("16", f"{totalnodes}"))
    elif "export MASTER_ADDR=$(hostname -i)" in line:
        continue
    elif "python train_basic.py" in line:
        slurm_script_content.append('declare -a adapcases=("ti_avit/baseline" "ti_svit/baseline" "ti_vit/baseline" "s_avit/baseline" "s_svit/baseline" "s_vit/baseline" "b_avit/baseline" "b_svit/baseline" "b_vit/baseline" "ti_svit/adap" "s_svit/adap" "b_svit/adap")\n')
        slurm_script_content.append('numcases=${#adapcases[@]}\n')
        slurm_script_content.append('NNODES=4\n')
        slurm_script_content.append('for (( i=0; i<${numcases}; i++ ));\n')
        slurm_script_content.append('do\n')
        slurm_script_content.append('  casedir=$PWD/${adapcases[$i]}\n')
        slurm_script_content.append('  TASK_ID=$i\n')
        slurm_script_content.append('  echo $casedir, $TASK_ID\n')
        slurm_script_content.append('  echo "srun -N$NNODES -n$((NNODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest -l python /lustre/orion/lrn037/scratch/zhangp/matey_updated/train_basic.py --run_name $run_name --config $config --yaml_config $casedir/$yaml_config --use_ddp > $casedir/log_$SLURM_JOB_ID.out &"\n')
        slurm_script_content.append('  srun -N$NNODES -n$((NNODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest -l python /lustre/orion/lrn037/scratch/zhangp/matey_updated/train_basic.py --run_name $run_name --config $config --yaml_config $casedir/$yaml_config --use_ddp > $casedir/log_$SLURM_JOB_ID.out &\n')
        slurm_script_content.append('  sleep 1\n')
        slurm_script_content.append('done\n')
        slurm_script_content.append('wait\n')
    else:
        slurm_script_content.append(line)

for ms in model_size:
    for modelatt in model_attention:
        pre_dir = os.path.join(base_dir, "%s_%s"%(ms, modelatt))
        os.makedirs(pre_dir, exist_ok=True)
        for adap in adaptivity:
            adapstr="adap" if adap else "baseline"
            if adap and modelatt!="svit":
                continue
            model_dir = os.path.join(pre_dir, adapstr)
            os.makedirs(model_dir, exist_ok=True)
                        
            with open("./config/%s"%config_src, "r") as file:
                template_lines = file.readlines()

            config_content = []
            for line in template_lines:
                if "exp_dir: './'" in line:
                    config_content.append(line.replace("exp_dir: './'", f"exp_dir: './{ms}_{modelatt}/{adapstr}'"))
                elif "learning_rate: -1" in line and setlr:
                    config_content.append(line.replace("learning_rate: -1", "learning_rate: 1e-4"))
                elif "  scheduler: 'cosine'" in line:
                    config_content.append("  scheduler: 'none'\n")
                elif "  epoch_size: 200" in line:
                    config_content.append(line.replace("  epoch_size: 200","  epoch_size: 100"))
                elif "  max_epochs: 120" in line:
                    config_content.append(line.replace("  max_epochs: 120","  max_epochs: 200"))
                elif " batch_size: 512 " in line:
                    config_content.append(line.replace("batch_size: 512","batch_size: 128"))
                elif "#gammaref: 0.2" in line and adap:
                    config_content.append(line.replace("#gammaref: 0.2", "gammaref: 0.4"))
                elif "patch_size: [[32, 32]]" in line and adap:
                    config_content.append(line.replace("patch_size: [[32, 32]]","patch_size: [[8, 8], [32, 32]]"))
                elif "embed_dim: 192" in line:
                    config_content.append(line.replace("embed_dim: 192","embed_dim: %d"%model_params[ms]["embed_dim"]))
                elif "num_heads: 3" in line:
                    config_content.append(line.replace("num_heads: 3","num_heads: %d"%model_params[ms]["num_heads"]))
                elif "model_type:" in line:
                    config_content.append("  model_type: '%s'\n"%(modelatt if modelatt!="vit" else "vit_all2all"))
                elif "time_type: 'attention'" in line:
                    if model_att_params[modelatt if modelatt!="vit" else "vit_all2all"] is None:
                        continue
                    elif "time_type" in model_att_params[modelatt]:
                        config_content.append("  time_type: '%s'\n"%model_att_params[modelatt]["time_type"])
                    else:
                        raise NotImplementedError
                elif "space_type: 'axial_attention'" in line:
                    if model_att_params[modelatt if modelatt!="vit" else "vit_all2all"] is None:
                        continue
                    elif "space_type" in model_att_params[modelatt]:
                        config_content.append("  space_type: '%s'\n"%model_att_params[modelatt]["space_type"])
                    else:
                        raise NotImplementedError
                elif "train_val_test: [.8, .1, .1]" in line:
                    continue
                    #startfrom = os.path.join("/lustre/orion/lrn037/scratch/zhangp/matey/",base_dir, "startpoint/%s_%s/baseline/basic_config/demo/training_checkpoints/ckpt.tar"%(ms, modelatt))
                    #config_content.append("  startfrom_path: '%s'\n"%startfrom)
                else:
                    config_content.append(line)

            config_file_path = os.path.join(model_dir, "config.yaml")
            with open(config_file_path, "w") as file:
                file.writelines(config_content)


            

with open(slurm_script_path, "w") as file:
    file.writelines(slurm_script_content)

bash_src = "submit_sequential_jobs.sh"
bash_path = os.path.join(base_dir, "submit_sequential_jobs.sh")
with open(bash_src, "r") as file:
    template_lines = file.readlines()

bash_script_content=[]
for line in template_lines:
    if "for i in $(seq 1 1);" in line:
        bash_script_content.append("for i in $(seq 1 2);\n")
    else:
        bash_script_content.append(line)

with open(bash_path, "w") as file:
    file.writelines(bash_script_content)

#os.system("bash submit_sequential_jobs.sh")
