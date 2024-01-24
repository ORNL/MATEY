import os
import glob, re
import random

leadtime_max=1
lr="1e-4"
base_dir = f"leadtimemax_{leadtime_max}_lr{lr}"
pretrainings=["avit_b92", "svit_b92", "vit_b92"]#, "vit_a92", "vit_b92s"]
model_types=["avit", "svit", "vit_all2all"]
finetunigs=["MW", "MW_all", "MW_INIT", "MW_INIT_all"]

model_params = {
    "avit": {"time_type": 'attention', "space_type": 'axial_attention'},
    "svit": {"time_type": 'all2all', "space_type": 'all2all'},
    "vit_all2all": None,
}

train_tot="/lustre/orion/lrn037/proj-shared/miniweather_thermals_small/train"
trainfiles_tot = glob.glob(train_tot + "/output1970994-*")
resampletrain = False
assert len(trainfiles_tot)==16

config_src ="finetuning_MW_data.yaml"
slurm_src = "submit_MW.sh"

config_count = 0
for idata in [1, 2, 4, 8, 16]:
    random.shuffle(trainfiles_tot)
    subtrain = os.path.join(train_tot, "train_%d"%idata)
    if resampletrain:
        os.makedirs(subtrain, exist_ok=True)
        for ifile in range(idata):
            file_exo=os.path.basename(trainfiles_tot[ifile])
            os.system("ln -s %s %s"%(os.path.join(train_tot, file_exo), os.path.join(subtrain, file_exo)))
    for pretrainmodel, modeltype in zip(pretrainings[1:], model_types[1:]):
        pre_dir = os.path.join(base_dir, pretrainmodel)
        os.makedirs(pre_dir, exist_ok=True)
        for finetunemod in finetunigs:
            #if "svit" in pre_dir and "all" in finetunemod:
            #    continue
            model_dir = os.path.join(pre_dir, finetunemod)
            os.makedirs(model_dir, exist_ok=True)
            imod = pretrainmodel.index("vit")
            init=""
            if "INIT" in finetunemod:
                init="_INIT"
            allstr=""
            if "all" in finetunemod:
                allstr="_all"
            
            with open("./config/%s"%config_src, "r") as file:
                template_lines = file.readlines()

            config_content = []
            for line in template_lines:
                if "exp_dir:" in line:
                    config_content.append(f"  exp_dir: ./\n")
                elif "model_type:" in line:
                    config_content.append(f"  model_type: {modeltype}\n")
                elif "time_type: 'attention'" in line:
                    if model_params[modeltype] is None:
                        continue
                    elif "time_type" in model_params[modeltype]:
                        config_content.append("  time_type: %s\n"%model_params[modeltype]["time_type"])
                    else:
                        raise NotImplementedError
                elif "space_type: 'axial_attention'" in line:
                    if model_params[modeltype] is None:
                        continue
                    elif "space_type" in model_params[modeltype]:
                        config_content.append("  space_type: %s\n"%model_params[modeltype]["space_type"])
                    else:
                        raise NotImplementedError
                elif "#pretrained_ckpt_path: 'INIT'" in line:
                    if "INIT" in finetunemod:
                        config_content.append(line.replace("#pretrained_ckpt_path: 'INIT'",f"pretrained_ckpt_path: 'INIT'"))
                    else:
                        config_content.append(line)
                elif "pretrained_ckpt_path: '/lustre/orion/" in line:
                    if "INIT" in finetunemod:
                        config_content.append(f"  #pretrained_ckpt_path: '/lustre/orion/lrn037/proj-shared/zhangp/portal/runs_pdebench_FM/{modeltype}_b92/val_logs/best_ckpt.tar'\n")
                    else:
                        if modeltype=="vit_all2all":
                            config_content.append(f"  pretrained_ckpt_path: '/lustre/orion/lrn037/proj-shared/zhangp/portal/runs_pdebench_FM/vit_b92/val_logs/best_ckpt.tar'\n")
                        else:
                            config_content.append(f"  pretrained_ckpt_path: '/lustre/orion/lrn037/proj-shared/zhangp/portal/runs_pdebench_FM/{modeltype}_b92/val_logs/best_ckpt.tar'\n")
                elif "freeze_middle: !!bool True" in line and "all" in finetunemod:
                    config_content.append("  freeze_middle: !!bool False\n")
                elif "/lustre/orion/lrn037/proj-shared/miniweather_thermals_small/train" in line:
                    config_content.append(line.replace("/lustre/orion/lrn037/proj-shared/miniweather_thermals_small/train",subtrain))
                else:
                    config_content.append(line)


            config_file_path = os.path.join(model_dir, "config.yaml")
            with open(config_file_path, "w") as file:
                file.writelines(config_content)

            slurm_script_path = os.path.join(model_dir, "submit.sh")
            with open(slurm_src, "r") as file:
                template_lines = file.readlines()

            slurm_script_content=[]
            for line in template_lines:
                if "#SBATCH -J avit_b92_MW" in line:
                    slurm_script_content.append("#SBATCH -J %s\n"%finetunemod)
                elif "python train_basic.py" in line:
                    slurm_script_content.append(line.replace("python train_basic.py","python  ./../../../train_basic.py"))
                else:
                    slurm_script_content.append(line)

            with open(slurm_script_path, "w") as file:
                file.writelines(slurm_script_content)

            os.chdir(model_dir)
            #os.system("bash submit_sequential_jobs.sh")
            os.chdir("/lustre/orion/lrn037/scratch/zhangp/matey")

            config_count += 1

print(f"{config_count} configuration files and SLURM scripts generated successfully.")
