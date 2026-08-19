# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import argparse
import os
import torch
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from matey import Inferencer
from matey.utils import setup_dist, YParams
import glob, socket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='./Dev_Fusion/basic_config/demo/', type=str)
    parser.add_argument("--yaml_config", default='hyperparams.yaml', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--on_perlmutter", action='store_true', help='Run inference on perlmutter')
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--leadtime", default=1, type=int)
    parser.add_argument("--AR", action='store_true', help='Use autoregressive rollout')
    parser.add_argument("--withoutXGC", action='store_true', help='No XGC data in the inference')

    args = parser.parse_args()
    params = YParams(os.path.join(args.model_dir, args.yaml_config))
    params.use_ddp = args.use_ddp
    params.autoregressive = args.AR
    if params.autoregressive:
      params.supportdata =[{"input_control_act": True}]
  

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

    #those are needed to mark the input/output heads
    """
    params.train_data_paths=[
              #[f'/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/solps/train', 'SOLPS2D', '','tk-2D'],
              [f'/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/solps/SOLPS2DwION/', 'SOLPS2DwION', '','tk-2D'],
              [f'/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/fusiond-seed-xgc1-data/', 'graphxgc', '', "tk-graph"]
              ]
    params.valid_data_paths=[
              #[f'/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/solps/valid', 'SOLPS2D', '','tk-2D'],
              [f'/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/solps/SOLPS2DwION/', 'SOLPS2DwION', '','tk-2D'],
              [f'/global/cfs/projectdirs/amsc007/zhan1668/MATEY/Datasets_pretraining/fusiond-seed-xgc1-data/', 'graphxgc', '', "tk-graph"]
              ]
    """
    params.train_data_paths = [
              ['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/train/', 'SOLPS2D', '','tk-2D'],
              ['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/SOLPS2DwION/', 'SOLPS2DwION', '','tk-2D'],
              ['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/gkeyll/', 'gkeylltcv', '', "tk-3D"],
              ['/lustre/orion/fus183/proj-shared/fusiond-seed-xgc1-data/', 'graphxgc', '', "tk-graph"]
              ]
    if args.withoutXGC:
       params.sp_groupsize = 1
       params.valid_data_paths = [
                ['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/valid/', 'SOLPS2D', '','tk-2D'],
                ['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/SOLPS2DwION/', 'SOLPS2DwION', '','tk-2D'],
                ['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/gkeyll/', 'gkeylltcv', '', "tk-3D"],
                #['/lustre/orion/fus183/proj-shared/fusiond-seed-xgc1-data/', 'graphxgc', '', "tk-graph"]
                ]
    else:
       if not hasattr(params, 'sp_groupsize') or params.sp_groupsize <=1:
          params.sp_groupsize = 8
          print(f"Warning : sp_groupsize is set to be 8/16 for XGC inference in current trained models but got no params.sp_groupsize for {args.model_dir}. Setting it to 8 for inference.")
       #assert hasattr(params, 'sp_groupsize') and params.sp_groupsize >1 , f"sp_groupsize is set to be 8/16 for XGC inference in current trained models but got {params.sp_groupsize}."
       params.valid_data_paths = [
                       #['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/valid/', 'SOLPS2D', '','tk-2D'],
                       #['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/SOLPS2DwION/', 'SOLPS2DwION', '','tk-2D'],
                       #['/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/gkeyll/', 'gkeylltcv', '', "tk-3D"],
                       ['/lustre/orion/fus183/proj-shared/fusiond-seed-xgc1-data/', 'graphxgc', '', "tk-graph"]
                       ]
    params.epoch_size = params.epoch_size*len(params.valid_data_paths) 
    inferencer = Inferencer(params, global_rank, local_rank, device)
    inferencer.inference()
    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
