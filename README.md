# MATEY ![MATEY](https://code.ornl.gov/uploads/-/system/group/avatar/24838/Matey_logo_single.png?width=48)

MATEY is a Multiscale AdapTivE trustworthY codebase for developing spatiotemporal foundation models of physcial systems.

## Installation
- Conda env installed on Frontier, `conda activate /lustre/orion/proj-shared/lrn037/gounley1/conda600whl`; see the usage example in `submit_batch.sh`
- Install your own virtual env
    * `python3.9 -m venv ~/virtual/matey`
    * `source ~/virtual/matey/bin/activate`
    * `pip install -r requirements_new.txt`
## Running
 
### Slurm Launch
- see the slurm job example, `submit_batch.sh`
### Single Device
- `python python train_basic.py "--run_name", "demo", "--config", "basic_config", "--yaml_config", "./config/mpp_avit_ti_config.yaml"`

## Publications & Presentations
- Pei Zhang, Paul Laiu, Matthew Norman, Doug Stefanski, and John Gounley, MATEY: multiscale adaptive foundation models for spatiotemporal physical systems, NeurIPS 2024 Workshop on  Machine Learning and the Physical Sciences. [accepted]  

## Contributors
We started our codebase in January 2024 from [PolymathicAI/multiple _physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) (with the commit: [67ffa35](https://github.com/PolymathicAI/multiple_physics_pretraining/tree/67ffa35acb087ed69b2c03b424dc2bf998b9ce0b)). Since then the codebase has been developed by the following contributors.

### Active Contributors

- John Gounley, ORNL
- Paul Laiu, ORNL
- Pei Zhang, ORNL

### Previous Contributors

- Sheikh Md Shakeel Hassan Nln (University of California, Irvine)
- Joseph Quinn (Vanderbilt University)





