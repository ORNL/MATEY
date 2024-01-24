import os


base_dir = "AMR4ViT_MW_Small"
model_attention=["avit", "svit", "vit"]
ps_params = {
    "ps_8": "[[8, 8]]",
    "ps_16": "[[16, 16]]",
    "ps_32": "[[32, 32]]",
    "ps_adap": "[[8, 8], [32, 32]]"
}

config_count = 0
for modelatt in model_attention[0:1]:
    pre_dir = os.path.join(base_dir, "%s"%modelatt)
    os.makedirs(pre_dir, exist_ok=True)

    

    for ps in ps_params:
        model_dir = os.path.join(pre_dir, ps)
        os.makedirs(model_dir, exist_ok=True)
        
        config_src = "mpp_%s_ti_config.yaml"%modelatt
        
        with open(base_dir+"/%s"%config_src, "r") as file:
            template_lines = file.readlines()

        config_content = []
        for line in template_lines:
            if "patch_size: [[8, 8], [32, 32]]" in line:
                config_content.append(line.replace("patch_size: [[8, 8], [32, 32]]","patch_size: %s"%ps_params[ps]))
            elif "sts_model: !!bool True" in line and ps!="ps_adap":
                config_content.append(line.replace("sts_model: !!bool True","sts_model: !!bool False"))
            else:
                config_content.append(line)
        config_file_path = os.path.join(model_dir, "config.yaml")
        with open(config_file_path, "w") as file:
            file.writelines(config_content)

        slurm_src = base_dir+"/submit_batch.sh"
        slurm_script_path = os.path.join(model_dir, "submit.sh")
        with open(slurm_src, "r") as file:
            template_lines = file.readlines()

        slurm_script_content=[]
        for line in template_lines:
            if "#SBATCH -J test" in line:
                slurm_script_content.append(line.replace("#SBATCH -J test","#SBATCH -J %s_%s"%(modelatt, ps)))
            elif "python train_basic.py" in line:
                slurm_script_content.append(line.replace("python train_basic.py","python ./../../../train_basic.py"))
            else:
                slurm_script_content.append(line)

        with open(slurm_script_path, "w") as file:
            file.writelines(slurm_script_content)

       
        os.chdir(model_dir)
        print("sbatch  %s" %slurm_script_path)
        #os.system("bash submit_sequential_jobs.sh")
        os.chdir("/lustre/orion/lrn037/scratch/zhangp/matey")

    config_count += 1

print(f"{config_count} configuration files and SLURM scripts generated successfully.")
