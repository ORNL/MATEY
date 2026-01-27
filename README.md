
# MATEY ![MATEY](https://github.com/ORNL/MATEY/blob/main/imgs/Matey_logo_single.png?raw=true?)
MATEY is a scalable open-source framework for developing transformer-based spatiotemporal foundation models for physical systems. It supports both structured and unstructured scientific datasets, providing multiscale representations, and enables efficient training on HPC systems. 

## Installation
- Option 1 - Frontier: a preconfigured Conda environment is available.
    * ROCM 6.3.1 
    * Activate the environment:
      ```bash 
      source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh
      ```
      
    * Example usage: `./examples/submit_JHTDB_demo.sh`
- Optional 2 - Create your own virtual env
    ```bash
    python3.9 -m venv ~/virtual/matey
    source ~/virtual/matey/bin/activate
    pip install -r requirements.txt
    ```
      
## Running
 
### Launching with Slurm
- Use the example Slurm job scripts inside `./examples`, e.g., 
  ```bash
  sbatch submit_batch.sh
  ```

## Training MATEY on Your Own Dataset

### Data loading
- Add your data loading script under: 

  `./matey/data_utils/<your_dataset_scripts>`

  See references:
  * `hdf5_3Ddatasets.py` 
  * `thewell_datasets.py` 
  * `netcdf_datasets.py`
- Register your dataset name in:

  `./matey/data_utils/dataset.py`->`DSET_NAME_TO_OBJECT`.
  
- Point your config file to the correct data directory 

### Model configuration
- Define your model architectures and data configurations in <your_config_yaml_file>.

  See examples in `./examples/config/Demo_*.ymal`.

### Submitting jobs
- Update <your_slurm_job_script> to include:

  ```bash
  export yaml_config=directory of <your_config_yaml_file>
  ```
- Submit your job:
  ```bash
  sbatch <your_slurm_job_script>
  ```  

## Publications & Presentations
- Hunor Csala, Sebastian De Pascuale, Paul Laiu, Jeremy Lore, Jae-Sun Park, Pei Zhang. Autoregressive long-horizon prediction of plasma edge dynamics. [arXiv:2512.23884](https://arxiv.org/abs/2512.23884)
- Junqi Yin, Mijanur Palash, M. Paul Laiu, Muralikrishnan Gopalakrishnan Meena, John Gounley, Stephen M. de Bruyn Kops, Feiyi Wang, Ramanan Sankaran, Pei Zhang. Pixel-Resolved Long-Context Learning for Turbulence at Exascale: Resolving Small-scale Eddies Toward the Viscous Limit. [arXiv:2507.16697](https://arxiv.org/abs/2507.16697)
- Pei Zhang, Paul Laiu, Matthew Norman, Doug Stefanski, and John Gounley. MATEY: multiscale adaptive foundation models for spatiotemporal physical systems. [arXiv:2412.20601](https://arxiv.org/abs/2412.20601)
- Pei Zhang, Paul Laiu, Matthew Norman, Doug Stefanski, and John Gounley. MATEY: multiscale adaptive foundation models for spatiotemporal physical systems, NeurIPS 2024 Workshop on  Machine Learning and the Physical Sciences. 

## Contributors
This codebase was originally seeded (Jan 2024) from [PolymathicAI/multiple _physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) (with the commit: [67ffa35](https://github.com/PolymathicAI/multiple_physics_pretraining/tree/67ffa35acb087ed69b2c03b424dc2bf998b9ce0b)). It has since been substantially rewritten and evolved independently, with ongoing development led by the following contributors.

### Active Contributors
- Hunor Csala, ORNL
- Andrey Prokopenko, ORNL
- Junqi Yin, ORNL
- Murali Gopalakrishnan Meena, ORNL
- John Gounley, ORNL
- Paul Laiu, ORNL
- Pei Zhang, ORNL

### Previous Contributors
- Mijanur R Palash, ORNL
- Xiao Jing (Georgia Tech; 2025 Summer Intern)
- Sheikh Md Shakeel Hassan Nln (University of California, Irvine; 2024 Summer Intern)
- Joseph Quinn (Vanderbilt University; 2024 Summer Intern)





