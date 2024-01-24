import os
import glob, re
import random


pretrained_dir="/lustre/orion/lrn037/scratch/zhangp/portal_finetuning/PDEBench_posbias"
pretrainings=["s_avit", "s_svit", "s_vit"]
model_types=["avit", "svit", "vit_all2all"]

finetunigs=["MHD", "MHD_all", "MHD_INIT", "MHD_INIT_all"]

lr="1e-4"
base_dir = "PDEBench_posbias"

model_params = {
    "avit": {"time_type": 'attention', "space_type": 'axial_attention'},
    "svit": {"time_type": 'all2all', "space_type": 'all2all'},
    "vit_all2all": None,
}
train_tot="/lustre/orion/lrn037/proj-shared/liquidMetalMHD/train"
trainfiles_tot = glob.glob(train_tot + "/*_sol.exo")

config_src ="finetuning_MHD_data.yaml"
slurm_src = "submit_MHD.sh"

resampletrain = False
assert len(trainfiles_tot)==24
config_count = 0
for idata in [1, 3, 6, 12, 24]:
    random.shuffle(trainfiles_tot)
    subtrain = os.path.join(train_tot, "train_%d"%idata)
    if resampletrain:
        os.makedirs(subtrain, exist_ok=True)
        for ifile in range(idata):
            file_exo=os.path.basename(trainfiles_tot[ifile])
            ist=re.search("ldc_Bx_", file_exo).start()+len("ldc_Bx_")
            Bx=file_exo[ist:ist+4]
            ist=re.search("_rem_", file_exo).start()+len("_rem_")
            Re=file_exo[ist:ist+3]
            file_numpy = f"numpy_Bx_{Bx}_Re_{Re}.npy"
            os.system("ln -s %s %s"%(os.path.join(train_tot, file_exo), os.path.join(subtrain, file_exo)))
            os.system("ln -s %s %s"%(os.path.join(train_tot, file_numpy), os.path.join(subtrain, file_numpy)))   
    
    case_dir = os.path.join(base_dir,"train_%d"%idata)
    for pretrainmodel, modeltype in zip(pretrainings[2:], model_types[2:]):
        fine_dir = os.path.join(case_dir,"train_%d"%idata, pretrainmodel)
        os.makedirs(fine_dir, exist_ok=True)
        for finetunemod in finetunigs:
            model_dir = os.path.join(fine_dir,  finetunemod)
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
                elif "/lustre/orion/lrn037/proj-shared/liquidMetalMHD/train" in line:
                    config_content.append(line.replace("/lustre/orion/lrn037/proj-shared/liquidMetalMHD/train",subtrain))
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
