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
from matey import Trainer
from matey.utils import setup_dist, check_sp, profile_function, log_to_file, log_versions, YParams
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='00', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--use_fsdp", action='store_true', help='Use FullyShardedDataParallel')
    parser.add_argument("--yaml_config", default='./config/multi_ds.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--pei_debug", action='store_true', help='Pei debugging flag')
    parser.add_argument("--pei_oneloss", action='store_true', help='Pei debugging flag')
    parser.add_argument("--pei_filtered", action='store_true', help='Pei filtering flag')
    parser.add_argument("--pei_minres", action='store_true', help='Pei minimize residual flag')
    parser.add_argument("--pei_moduleloss", action='store_true', help='Pei optimize module loss flag')
    parser.add_argument("--pei_fixedupsample", action='store_true', help='Pei fix upsampling flag')
    parser.add_argument("--pei_linearupsample", action='store_true', help='Pei linear upsampling flag')
    parser.add_argument("--enable_sync", action='store_true', help='torch.cuda.synchronize flag')
    parser.add_argument("--enable_profiling", action='store_true', help='enable torch profiler flag')

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = args.use_ddp
    params.use_fsdp = args.use_fsdp
    params.pei_debug = args.pei_debug
    params.pei_oneloss = args.pei_oneloss
    params.pei_filtered = args.pei_filtered
    params.pei_minres = args.pei_minres
    params.pei_moduleloss = args.pei_moduleloss
    params.enable_sync = args.enable_sync   
    params.profiling =  args.enable_profiling 

    if not hasattr(params, "tokenizer_heads"):
        assert hasattr(params, "patch_size")
        params.tokenizer_heads=[{"head_name": "default",
                                 "patch_size": params.patch_size 
                                 }]
    print(params.tokenizer_heads, flush=True)
    if hasattr(params, "hierarchical"):
        params.hierarchical["fixedupsample"] =args.pei_fixedupsample
        params.hierarchical["linearupsample"]=args.pei_linearupsample
        print(params.hierarchical, flush=True)
    # Set up distributed training
    device, world_size, local_rank, global_rank = setup_dist(params)
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}", flush=True)

    # Modify params
    params['batch_size'] =int(params.batch_size//world_size)
    params['startEpoch'] = 0
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))

    params['old_exp_dir'] = expDir # I dont remember what this was for but not removing it yet
    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['old_checkpoint_path'] = os.path.join(params.old_exp_dir, 'training_checkpoints/best_ckpt.tar')

    # Have rank 0 check for and/or make directory
    if  global_rank==0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)
    if params.use_fsdp:
        params['resuming'] = True if len(glob.glob(os.path.join(params.best_checkpoint_path, "*distcp")))>0 else False
    else:
        params['resuming'] = True if os.path.isfile(params.best_checkpoint_path) else False

    if params.pei_debug:
        params.debug_outdir = os.path.join(expDir, "./debug_outputs/")
        os.makedirs(params.debug_outdir, exist_ok=True)

    if global_rank==0:
        log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        log_versions()
        params.log()

    params['log_to_screen'] = (global_rank==0) and params['log_to_screen']
    torch.backends.cudnn.benchmark = False

    if global_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile )
    trainer = Trainer(params, global_rank, local_rank, device)

    #check if groups are defined properly
    if hasattr(params, "sp_groupsize"):
        check_sp(trainer.sequence_parallel_groups, global_rank)
    
    with profile_function(enabled=trainer.profiling, logdir="./log_profiler_section_8worker_pinmem_prefetch_factor2") as prof:
        trainer.train()
        prof.step()
    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
