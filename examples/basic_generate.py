import argparse
import os
import torch
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import argparse
import os
import torch
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from matey import Generator
from matey.utils import setup_dist, check_sp, profile_function, log_to_file, log_versions, YParams
import glob, socket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='./Demo_Diffusion_CIFAR10_3level_B_1e-3lr/basic_config/demo_diffusion/', type=str)
    parser.add_argument("--yaml_config", default='hyperparams.yaml', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--output_dir", default='./CIFAR10_generation_outputs/', type=str)

    args = parser.parse_args()
    params = YParams(os.path.join(args.model_dir, args.yaml_config))
    params.use_ddp = args.use_ddp
    params['output_dir'] = args.output_dir

    os.makedirs(params.output_dir, exist_ok=True)
    
    # Set up distributed training
    device, world_size, local_rank, global_rank = setup_dist(params)
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}, host={socket.gethostname()}", flush=True)

    # Modify params
    params['batch_size'] =int(params.batch_size//world_size)
    params['checkpoint_path'] = os.path.join(args.model_dir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(args.model_dir, 'training_checkpoints/best_ckpt.tar')

    assert os.path.isfile(params.checkpoint_path), f"file {params.checkpoint_path} not found" 
    assert os.path.isfile(params.best_checkpoint_path), f"file {params.best_checkpoint_path} not found" 
    params['resuming'] = True 

    generator = Generator(params, global_rank, local_rank, device)

    generator.generate(seed=42, num_samples=9, batch_list=[6])

    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
