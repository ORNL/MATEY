import os
import glob, re
import random


pretrained_dir="/lustre/orion/lrn037/scratch/zhangp/portal_finetuning/PDEBench_posbias"
pretrainings=["s_avit", "s_svit", "s_vit"]
model_att_types=["avit", "svit", "vit_all2all"]
pretadap="baseline"

base_dir = "PDEBench_posbias_finetuning"

model_att_params = {
    "avit": {"time_type": 'attention', "space_type": 'axial_attention'},
    "svit": {"time_type": 'all2all', "space_type": 'all2all'},
    "vit_all2all": None,
}

model_params = {
    "ti": {"embed_dim": 192, "num_heads": 3},
    "s": {"embed_dim": 384, "num_heads": 6},
    "b": {"embed_dim": 768, "num_heads": 12},
}

finestrategies=["prepost", "all", "prepost_INIT", "all_INIT"]

finetuning_cases = {
    "MW": {
        "train_tot": "/lustre/orion/lrn037/proj-shared/miniweather_thermals_small/train", 
        "folder_spec": "/output1970994-*",
        "data_cases": [1, 2, 4, 8, 16],
        "config_src":"finetuning_MW_data.yaml",
        "slurm_src":"submit_finetuning.sh",
        },
    "MHD": {
        "train_tot": "/lustre/orion/lrn037/proj-shared/liquidMetalMHD/train",
        "folder_spec": "/*_sol.exo",
        "data_cases": [1, 3, 6, 12, 24],
        "config_src":"finetuning_MHD_data.yaml",
        "slurm_src":"submit_finetuning.sh",
        },
}

resampletrain = False

for finecase in finetuning_cases:

    train_tot=finetuning_cases[finecase]["train_tot"]
    folder_spec=finetuning_cases[finecase]["folder_spec"]
    data_cases=finetuning_cases[finecase]["data_cases"]
    config_src=finetuning_cases[finecase]["config_src"]
    slurm_src=finetuning_cases[finecase]["slurm_src"]

    trainfiles_tot = glob.glob(train_tot + folder_spec)
    assert len(trainfiles_tot)==data_cases[-1]

    finecase_parent = os.path.join(base_dir, finecase)

    for idata in data_cases:
        random.shuffle(trainfiles_tot)
        subtrain = os.path.join(train_tot, "train_%d"%idata)
        if resampletrain:
            os.makedirs(subtrain, exist_ok=True)
            if finecase=="MHD":
                for ifile in range(idata):
                    file_exo=os.path.basename(trainfiles_tot[ifile])
                    ist=re.search("ldc_Bx_", file_exo).start()+len("ldc_Bx_")
                    Bx=file_exo[ist:ist+4]
                    ist=re.search("_rem_", file_exo).start()+len("_rem_")
                    Re=file_exo[ist:ist+3]
                    file_numpy = f"numpy_Bx_{Bx}_Re_{Re}.npy"
                    os.system("ln -s %s %s"%(os.path.join(train_tot, file_exo), os.path.join(subtrain, file_exo)))
                    os.system("ln -s %s %s"%(os.path.join(train_tot, file_numpy), os.path.join(subtrain, file_numpy)))  
            else:
                for ifile in range(idata):
                    file_exo=os.path.basename(trainfiles_tot[ifile])
                    os.system("ln -s %s %s"%(os.path.join(train_tot, file_exo), os.path.join(subtrain, file_exo)))
    
        case_dir = os.path.join(finecase_parent,"train_%d"%idata)
        for pretrainmodel, modelatttype in zip(pretrainings[2:], model_att_types[2:]):
            imod = pretrainmodel.index("_")
            modelsize=pretrainmodel[:imod]
            #for tunestg in finestrategies[::2]:
            for tunestg in finestrategies[1::2]:

                model_dir = os.path.join(case_dir, "%s_%s"%(pretrainmodel, tunestg))
                os.makedirs(model_dir, exist_ok=True)
                
                with open("./config/%s"%config_src, "r") as file:
                    template_lines = file.readlines()

                config_content = []
                for line in template_lines:
                    if "embed_dim: 192" in line:
                        config_content.append(line.replace("embed_dim: 192","embed_dim: %d"%model_params[modelsize]["embed_dim"]))
                    elif "num_heads: 3" in line:
                        config_content.append(line.replace("num_heads: 3","num_heads: %d"%model_params[modelsize]["num_heads"]))
                    elif "model_type:" in line:
                        config_content.append(f"  model_type: {modelatttype}\n")
                    elif "time_type: 'attention'" in line:
                        if model_att_params[modelatttype] is None:
                            continue
                        elif "time_type" in model_att_params[modelatttype]:
                            config_content.append("  time_type: %s\n"%model_att_params[modelatttype]["time_type"])
                        else:
                            raise NotImplementedError
                    elif "space_type: 'axial_attention'" in line:
                        if model_att_params[modelatttype] is None:
                            continue
                        elif "space_type" in model_att_params[modelatttype]:
                            config_content.append("  space_type: %s\n"%model_att_params[modelatttype]["space_type"])
                        else:
                            raise NotImplementedError
                    elif "#pretrained_ckpt_path: 'INIT'" in line:
                        if "INIT" in tunestg:
                            config_content.append(line.replace("#pretrained_ckpt_path: 'INIT'",f"pretrained_ckpt_path: 'INIT'"))
                        else:
                            config_content.append(line)
                    elif "  pretrained_ckpt_path: 'pretrainedtar'" in line:
                        if "INIT" in tunestg:
                            config_content.append(f"  #pretrained_ckpt_path: 'pretrainedtar'\n")
                        else:
                            config_content.append(f"  pretrained_ckpt_path: '{pretrained_dir}/{pretrainmodel}/{pretadap}/val_logs/best_ckpt.tar'\n")
                    elif "freeze_middle: !!bool True" in line and "all" in tunestg:
                        config_content.append("  freeze_middle: !!bool False\n")
                    elif "finetuningtraindata_dir" in line:
                        config_content.append(line.replace("finetuningtraindata_dir",subtrain))
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
                    if "#SBATCH -J ft" in line:
                        slurm_script_content.append("#SBATCH -J %s-%s\n"%(pretrainmodel, tunestg))
                    elif "python train_basic.py" in line:
                        slurm_script_content.append(line.replace("python train_basic.py","python  /lustre/orion/lrn037/scratch/zhangp/matey/train_basic.py"))
                    else:
                        slurm_script_content.append(line)

                with open(slurm_script_path, "w") as file:
                    file.writelines(slurm_script_content)

                os.chdir(model_dir)
                os.system("sbatch submit.sh")
                os.chdir("/lustre/orion/lrn037/scratch/zhangp/matey")

