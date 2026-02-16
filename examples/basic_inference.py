# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

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
from matey import Inferencer
from matey.utils import setup_dist YParams
import glob, socket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='./Dev_Fusion/basic_config/demo/', type=str)
    parser.add_argument("--yaml_config", default='hyperparams.yaml', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--config", default='basic_config', type=str)

    args = parser.parse_args()
    params = YParams(os.path.join(args.model_dir, args.yaml_config))
    params.use_ddp = args.use_ddp
  

    # Set up distributed training
    device, world_size, local_rank, global_rank = setup_dist(params)
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}, host={socket.gethostname()}", flush=True)

    # Modify params
    params['batch_size'] =1
    params['checkpoint_path'] = os.path.join(args.model_dir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(args.model_dir, 'training_checkpoints/best_ckpt.tar')

    assert os.path.isfile(params.checkpoint_path), f"file {params.checkpoint_path} not found" 
    assert os.path.isfile(params.best_checkpoint_path), f"file {params.best_checkpoint_path} not found" 
    params['resuming'] = True 

    inferencer = Inferencer(params, global_rank, local_rank, device)
    inferencer.inference()

    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
