import os


base_dir = "AMR4ViT_MW"
model_attention=["avit", "svit", "vit"]
model_size=["ti","s","b"]
adaptivity=[True, False]

model_params = {
    "ti": {"embed_dim": 192, "num_heads": 3},
    "s": {"embed_dim": 384, "num_heads": 6},
    "b": {"embed_dim": 768, "num_heads": 12},
}

config_count = 0
for ms in model_size:
    for modelatt in model_attention:
        pre_dir = os.path.join(base_dir, "%s_%s"%(ms, modelatt))
        os.makedirs(pre_dir, exist_ok=True)
        for adap in adaptivity:
            adapstr="adap" if adap else "baseline"
            model_dir = os.path.join(pre_dir, adapstr)
            os.makedirs(model_dir, exist_ok=True)
            
            config_src = "MW_%s_ti_baseline.yaml"%modelatt
            
            with open("./config/%s"%config_src, "r") as file:
                template_lines = file.readlines()

            config_content = []
            for line in template_lines:
                if "#gammaref: 0.2" in line and adap:
                    config_content.append(line.replace("#gammaref: 0.2", "gammaref: 0.2"))
                elif "patch_size: [[32, 32]]" in line and adap:
                    config_content.append(line.replace("patch_size: [[32, 32]]","patch_size: [[8, 8], [32, 32]]"))
                elif "embed_dim: 192" in line:
                    config_content.append(line.replace("embed_dim: 192","embed_dim: %d"%model_params[ms]["embed_dim"]))
                elif "num_heads: 3" in line:
                    config_content.append(line.replace("num_heads: 3","num_heads: %d"%model_params[ms]["num_heads"]))
                else:
                    config_content.append(line)
            config_file_path = os.path.join(model_dir, "config.yaml")
            with open(config_file_path, "w") as file:
                file.writelines(config_content)

            slurm_src = "submit_MW_AMR4ViT.sh"
            slurm_script_path = os.path.join(model_dir, "submit.sh")
            with open(slurm_src, "r") as file:
                template_lines = file.readlines()

            slurm_script_content=[]
            for line in template_lines:
                if "job_MW" in line:
                    slurm_script_content.append(line.replace("job_MW","%s_%s_%s"%(ms, modelatt, adapstr)))
                elif "python train_basic.py" in line:
                    slurm_script_content.append(line.replace("python train_basic.py","python ./../../../train_basic.py"))
                else:
                    slurm_script_content.append(line)

            with open(slurm_script_path, "w") as file:
                file.writelines(slurm_script_content)

            bash_src = "submit_sequential_jobs.sh"
            bash_path = os.path.join(model_dir, "submit_sequential_jobs.sh")
            with open(bash_src, "r") as file:
                template_lines = file.readlines()
            bash_script_content=[]
            for line in template_lines:
                if "for i in $(seq 1 1);" in line:
                    bash_script_content.append("for i in $(seq 1 3);\n")
                else:
                    bash_script_content.append(line)

            with open(bash_path, "w") as file:
                file.writelines(bash_script_content)

            os.chdir(model_dir)
            print("bash %s" %bash_path)
            os.system("bash submit_sequential_jobs.sh")
            os.chdir("/lustre/orion/lrn037/scratch/zhangp/matey")

            config_count += 1

print(f"{config_count} configuration files and SLURM scripts generated successfully.")
