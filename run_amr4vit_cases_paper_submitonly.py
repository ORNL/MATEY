import os


base_dir = "AMR4ViT_MW_posbias"

base_dir = "AMR4ViT_MW_posbias_lr1e-4"
setlr=True

model_attention=["avit", "svit", "vit"]
model_size=["ti","s"] #,"b"]
adaptivity=[True, False]

model_params = {
    "ti": {"embed_dim": 192, "num_heads": 3},
    "s": {"embed_dim": 384, "num_heads": 6},
    "b": {"embed_dim": 768, "num_heads": 12},
}
model_att_params = {
    "avit": {"time_type": 'attention', "space_type": 'axial_attention'},
    "svit": {"time_type": 'all2all', "space_type": 'all2all'},
    "vit_all2all": None,
}

config_src ="MW_baseline.yaml"
slurm_src = "submit_MW_AMR4ViT.sh"


config_count = 0
for ms in model_size:
    for modelatt in model_attention:
        pre_dir = os.path.join(base_dir, "%s_%s"%(ms, modelatt))
        os.makedirs(pre_dir, exist_ok=True)
        for adap in adaptivity:
            adapstr="adap" if adap else "baseline"
            model_dir = os.path.join(pre_dir, adapstr)

            bash_src = "submit_sequential_jobs.sh"
            bash_path = os.path.join(model_dir, "submit_sequential_jobs.sh")
            with open(bash_src, "r") as file:
                template_lines = file.readlines()
            bash_script_content=[]
            for line in template_lines:
                if "for i in $(seq 1 1);" in line:
                    bash_script_content.append("for i in $(seq 1 6);\n")
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
