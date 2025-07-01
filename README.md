# MATEY ![MATEY](https://code.ornl.gov/uploads/-/system/group/avatar/24838/Matey_logo_single.png?width=48)

MATEY is a Multiscale AdapTivE trustworthY codebase for developing spatiotemporal foundation models of physcial systems.

## Installation
- Running jobs on Frontier (Conda env installed)
    * ROCM 6.3.1 (recommended), `source /lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh`; see the usage example in ``
    * ROCM 6.0.0 (outdated), `conda activate /lustre/orion/proj-shared/lrn037/gounley1/conda600whl`; see the usage example in `submit_batch.sh`
    * ROMC 6.0.0 (world-shared) (outdated), `source /lustre/orion/world-shared/lrn037/gounley1/env600.sh`; see the usage example in `submit_batch.sh`
- Install your own virtual env
    * `python3.9 -m venv ~/virtual/matey`
    * `source ~/virtual/matey/bin/activate`
    * `pip install -r requirements.txt`
## Running
 
### Slurm Launch
- see the slurm job example, `sbatch submit_batch.sh`
### Single Device
- `python train_basic.py "--run_name", "demo", "--config", "basic_config", "--yaml_config", "./config/mpp_avit_ti_config.yaml"`

## Publications & Presentations
- Pei Zhang, Paul Laiu, Matthew Norman, Doug Stefanski, and John Gounley, MATEY: multiscale adaptive foundation models for spatiotemporal physical systems. [arXiv:2412.20601](https://arxiv.org/abs/2412.20601)
- Pei Zhang, Paul Laiu, Matthew Norman, Doug Stefanski, and John Gounley, MATEY: multiscale adaptive foundation models for spatiotemporal physical systems, NeurIPS 2024 Workshop on  Machine Learning and the Physical Sciences. [accepted]  

## Contributors
This code structure was originally seeded in January 2024 from [PolymathicAI/multiple _physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) (with the commit: [67ffa35](https://github.com/PolymathicAI/multiple_physics_pretraining/tree/67ffa35acb087ed69b2c03b424dc2bf998b9ce0b)). Since then the codebase has been substantially rewritten and evolved independently, with ongoing development led by the following contributors.

### Active Contributors

- Hunor Csala, ORNL
- Andrey Prokopenko, ORNL
- Junqi Yin, ORNL
- Mijanur R Palash, ORNL
- Murali Gopalakrishnan Meena, ORNL
- John Gounley, ORNL
- Paul Laiu, ORNL
- Pei Zhang, ORNL

### Previous Contributors

- Sheikh Md Shakeel Hassan Nln (University of California, Irvine)
- Joseph Quinn (Vanderbilt University)





