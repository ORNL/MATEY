import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict
import pickle as pkl
from torchinfo import summary
try:
    from data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
    from models.avit import build_avit
    from utils import logging_utils
    from utils.YParams import YParams
except:
    from .data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
    from .models.avit import build_avit
    from .utils import logging_utils
    from .utils.YParams import YParams
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Inferencer:
    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.log_to_screen = params.log_to_screen
        # Basic setup
        self.model_dir=params.experiment_dir
        self.output_dir=params.output

        self.initialize_data(self.params)
        print(f"Initializing model on rank {self.global_rank}")
        self.initialize_model(self.params)
        print("Loading checkpoint %s"%params.checkpoint_path)
        self.load_model_checkpoint(params.checkpoint_path)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(' '.join([str(t) for t in text]))

    def initialize_data(self, params):
        if params.tie_batches:
            in_rank = 0
        else:
            in_rank = self.global_rank
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")
        _, self.train_dataset, _ = get_data_loader(params, params.train_data_paths,
                                                                        dist.is_initialized(), split='train', rank=in_rank, 
                                                                        train_offset=self.params.embedding_offset)
        _, self.valid_dataset, _ = get_data_loader(params, params.valid_data_paths,
                                                                        dist.is_initialized(), split='val', rank=in_rank)
        _,  self.test_dataset, _ = get_data_loader(params, params.test_data_paths,
                                                                        dist.is_initialized(),split='test', rank=in_rank)
    def initialize_model(self, params):
        if self.params.model_type == 'avit':
            self.model = build_avit(params).to(device)
        
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                 output_device=[self.local_rank], find_unused_parameters=True)
        
        self.single_print(f'Model parameter count: {sum([p.numel() for p in self.model.parameters()])}')

   
    def load_model_checkpoint(self, checkpoint_path):
        """ Load model/opt from path """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else torch.device('cpu'))
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        else:
            model_state = checkpoint
        try: # Try to load with DDP Wrapper
            self.model.load_state_dict(model_state)
        except: # If that fails, either try to load into module or strip DDP prefix
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(model_state)
            else:
                new_state_dict = OrderedDict()
                for key, val in model_state.items():
                    # Failing means this came from DDP - strip the DDP prefix
                    name = key[7:]
                    new_state_dict[name] = val
                self.model.load_state_dict(new_state_dict)
        checkpoint = None
        self.model = self.model.to(self.device)

    def plot_visual_contourcomp(self, x_true, x, var_names, outputname, iplot=3):
        T, C, H, W = x.shape        
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        for iplot in range(C):
            ax=axs[0, iplot]
            im=ax.contourf(x_true[0,iplot,:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(var_names[iplot])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
        
            ax =axs[1, iplot]
            im=ax.contourf(x[0,iplot,:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')

        #fig.tight_layout()
        plt.savefig(outputname)
        plt.close()

    def plot_visual_contourcompthree(self, x_true, x, x_auto, var_names, outputname, iplot=3, lt=1, x_inp=None):
        T, C, H, W = x.shape   
        if x_inp is None:     
            fig, axs = plt.subplots(3, 4, figsize=(20, 12))
            inp=-1
        else:
            fig, axs = plt.subplots(4, 4, figsize=(20, 16))
            inp=0


        for iplot in range(C):
            if inp==0:
                ax=axs[inp, iplot]
                im=ax.contourf(x_inp[0, -1, iplot,:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title(var_names[iplot], fontsize=18)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, orientation='vertical')
                if iplot==0:
                    ax.text(-100,100, "Input[-1]", fontsize=18)

            ax=axs[0+inp+1, iplot]
            im=ax.contourf(x_true[0,iplot,:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            ax.axis('off')
            if inp<0:
                ax.set_title(var_names[iplot], fontsize=18)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
            if iplot==0:
                ax.text(-100,100, "True at %d"%lt, fontsize=18)
        
            ax =axs[1+inp+1, iplot]
            im=ax.contourf(x[0,iplot,:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
            if iplot==0:
                ax.text(-120,100, "Train-LT%d"%self.params.train_leadtime_max, fontsize=18)

            ax =axs[2+inp+1, iplot]
            im=ax.contourf(x_auto[0,iplot,:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
            ax.set_aspect('equal')
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical')
            if iplot==0:
                ax.text(-120,100, "Auto-regress", fontsize=18)

        #fig.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.05, right=0.975, top=0.95, wspace=0.1, hspace=0.05)
        plt.savefig(outputname)
        plt.close()

    def plot_visual_contourindivi(self, x, outputname):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im=ax.contourf(x[:,:].squeeze().detach().numpy(), cmap="jet", levels=50)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout()
        plt.savefig(outputname)
        plt.close()

    def plot_loss_leadingtime(self, leadtime, samp, err_model_autoreg, err_model_LeadTim, err_reference, outputname):
        samp, err_model_autoreg, err_model_LeadTim, err_reference  = zip(*sorted(zip(samp, err_model_autoreg, err_model_LeadTim, err_reference)))
        print(leadtime, samp, err_model_autoreg, err_model_LeadTim, err_reference)
        colorlib=["k","b","g","r","m","y"]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for isample in range(len(err_reference)):
            ax.plot(leadtime, err_model_LeadTim[isample], colorlib[isample%len(colorlib)]+"-",label="%d-Train-LT%d"%(samp[isample],self.params.train_leadtime_max))
            ax.plot(leadtime, err_model_autoreg[isample], colorlib[isample%len(colorlib)]+"--", label="%d-Auto-regress"%samp[isample])
            ax.plot(leadtime, err_reference[isample], colorlib[isample%len(colorlib)]+":", label="%d-Ref."%samp[isample])
            print(leadtime,err_reference[isample] )
        ax.legend(bbox_to_anchor=(1.0, 0.7))
        ax.set_title("Prediction loss vs. leading time")
        ax.set_xlabel("Leadtime")
        ax.set_ylabel("Normalized RMSE Loss")
        fig.tight_layout()
        plt.savefig(outputname)
        plt.close()
                        
    def loss_calculate(self, tar, output, dset_type, subset, field_labels, dataset="valid"):
        logs={}
        #FIXME: double check their loss definition; spatial_dims is not actually spatial dim [Pei]
        spatial_dims = tuple(range(output.ndim))[2:] # Assume 0, 1, 2 are T, B, C
        residuals = output - tar
        nmse = (residuals.pow(2).mean(spatial_dims, keepdim=True) 
                / (1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True))).sqrt()#.mean()
        logs[f'{dset_type}/{dataset}_nrmse'] = logs.get(f'{dset_type}/{dataset}_nrmse',0) + nmse.mean()
        logs[f'{dset_type}/{dataset}_rmse'] = (logs.get(f'{dset_type}/{dataset}_rmse',0) 
                                            + residuals.pow(2).mean(spatial_dims).sqrt().mean())
        logs[f'{dset_type}/{dataset}_l1'] = (logs.get(f'{dset_type}/{dataset}_l1', 0) 
                                            + residuals.abs().mean())

        for i, field in enumerate(self.valid_dataset.subset_dict[subset.type]):
            field_name = field_labels[field]
            logs[f'{dset_type}/{field_name}_{dataset}_nrmse'] = (logs.get(f'{dset_type}/{field_name}_{dataset}_nrmse', 0) 
                                                            + nmse[:, i].mean())
            logs[f'{dset_type}/{field_name}_{dataset}_rmse'] = (logs.get(f'{dset_type}/{field_name}_{dataset}_rmse', 0) 
                                                                + residuals[:, i:i+1].pow(2).mean(spatial_dims).sqrt().mean())
            logs[f'{dset_type}/{field_name}_{dataset}_l1'] = (logs.get(f'{dset_type}/{field_name}_{dataset}_l1', 0) 
                                                                            +  residuals[:, i].abs().mean())
        
        logs_config = {}
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                logs_config[k] = v.item()
            else:
                logs_config[k] = v
        return logs_config
    
    def inference_extra(self, nsamples=5, LT=100):
        self.model.eval()
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Inference...")
        
        with torch.inference_mode():
            for setname, dataset in zip(["valid", "test"],[self.valid_dataset, self.test_dataset]):
                field_labels = dataset.get_state_names()
                # Iterate through all folder specific datasets
                for subset_group in dataset.sub_dsets:
                    for subset in subset_group.get_per_file_dsets():
                        dset_type = subset.title
                        self.single_print('Inferencing ON', setname, dset_type)
                        # Create data loader for each
                        sampleids=torch.randint(1, len(subset), (nsamples,))
                        err_model_LeadTim = [[0]*LT for _ in range(nsamples)]
                        err_model_autoreg = [[0]*LT for _ in range(nsamples)]
                        #use the difference between last snapshot of input and tar as error reference
                        err_reference = [[0]*LT for _ in range(nsamples)]
                        for isamp in range(nsamples):
                            isample = sampleids[isamp].item()
                            print(isample)
                            for it in range(1, LT+1):
                                index=[isample, it]
                                data = subset[index]
                                try:
                                    inp, bcs, tar, leadtime = map(lambda x: torch.tensor(x).unsqueeze(0).to(self.device), data) 
                                    refineind = None
                                except:
                                    inp, bcs, tar, refineind, leadtime = map(lambda x: torch.tensor(x).unsqueeze(0).to(self.device), data) 
                                if leadtime!=it:
                                    print("leadtime %d is beyond file length, change to %d instead"%(it, leadtime))
                                # Labels come from the trainset - useful to configure an extra field for validation sets not included
                                labels = torch.tensor(self.train_dataset.subset_dict.get(subset.get_name(), [-1]*len(dataset.subset_dict[subset.get_name()])),
                                                    device=self.device).unsqueeze(0).expand(tar.shape[0], -1)
                                inp = rearrange(inp, 'b t c h w -> t b c h w')
                                output_leadtime = self.model(inp, labels, bcs, leadtime=leadtime, refineind=refineind)
                                inp0 = inp
                                for _ in range(int(leadtime[0,0])):
                                    output_auto = self.model(inp0, labels, bcs, leadtime=torch.ones_like(leadtime), refineind=refineind)
                                    inp1=inp0.clone()
                                    inp1[:-1]=inp0[1:]
                                    inp1[-1]=output_auto
                                    inp0 = inp1
                                subset.get_min_max()
                                log_out=self.loss_calculate(tar, output_leadtime, dset_type, subset, field_labels, dataset=setname)
                                log_out_auto=self.loss_calculate(tar, output_auto, dset_type, subset, field_labels, dataset=setname)
                                log_reference=self.loss_calculate(tar, inp[:,-1,:,:,:], dset_type, subset, field_labels, dataset=setname)
                                print(f"For sample {isample} in {setname}, the error between auto and leadtime {leadtime} is", log_out_auto[f'{dset_type}/{setname}_nrmse'], log_out[f'{dset_type}/{setname}_nrmse'])
                                err_model_LeadTim[isamp][it-1]=log_out[f'{dset_type}/{setname}_nrmse']
                                err_model_autoreg[isamp][it-1]=log_out_auto[f'{dset_type}/{setname}_nrmse']
                                err_reference[isamp][it-1]=log_reference[f'{dset_type}/{setname}_nrmse']
                        filename = os.path.join(self.output_dir,f"./loss_vs_LT_{setname}.png")
                        self.plot_loss_leadingtime(np.arange(1,LT+1), [samp.item() for samp in sampleids],err_model_autoreg, err_model_LeadTim, err_reference, filename)              
            sys.exit(0)
    def inference(self, nsamples=5):
        self.model.eval()
        if self.global_rank == 0:
            summary(self.model)
        self.single_print("Starting Inference...")
        
        with torch.inference_mode():
            for setname, dataset in zip(["valid", "test"],[self.valid_dataset, self.test_dataset]):
                field_labels = dataset.get_state_names()
                # Iterate through all folder specific datasets
                for subset_group in dataset.sub_dsets:
                    for subset in subset_group.get_per_file_dsets():
                        dset_type = subset.title
                        self.single_print('Inferencing ON', setname, dset_type)
                        # Create data loader for each
                        sampleids=torch.randint(1, len(subset), (nsamples,))
                        for isamp in range(nsamples):
                            isample = sampleids[[isamp]]
                            data = subset[isample]
                            # Only do a few batches of each dataset if not doing full validation
                            #inp, bcs, tar = map(lambda x: x.to(self.device), data) 
                            try:
                                inp, bcs, tar, leadtime = map(lambda x: torch.tensor(x).unsqueeze(0).to(self.device), data) 
                                refineind = None
                            except:
                                inp, bcs, tar, refineind, leadtime = map(lambda x: torch.tensor(x).unsqueeze(0).to(self.device), data) 
                            
                            # Labels come from the trainset - useful to configure an extra field for validation sets not included
                            labels = torch.tensor(self.train_dataset.subset_dict.get(subset.get_name(), [-1]*len(dataset.subset_dict[subset.get_name()])),
                                                device=self.device).unsqueeze(0).expand(tar.shape[0], -1)
                            inp = rearrange(inp, 'b t c h w -> t b c h w')
                            output_leadtime = self.model(inp, labels, bcs, leadtime=leadtime, refineind=refineind)
                            inp0 = inp
                            for iT in range(int(leadtime[0,0])):
                                output_auto = self.model(inp0, labels, bcs, refineind=refineind)
                                inp1=inp0.clone()
                                inp1[:-1]=inp0[1:]
                                inp1[-1]=output_auto
                                inp0 = inp1

                            labelsplot = labels.tolist()[0]
                            print(labelsplot)
                            subset.get_min_max()
                            tarplot=tar.clone()
                            outputplot=output_leadtime.clone()
                            output_autoplot=output_auto.clone()
                            log_out=self.loss_calculate(tar, output_leadtime, dset_type, subset, field_labels, dataset=setname)
                            log_out_auto=self.loss_calculate(tar, output_auto, dset_type, subset, field_labels, dataset=setname)
                            print(f"For sampel {isample} in {setname}, the error between auto and leadtime {leadtime} is", log_out_auto[f'{dset_type}/{setname}_nrmse'], log_out[f'{dset_type}/{setname}_nrmse'])
                        
                            

                            if True:
                                for ivar, ilabel in enumerate(labelsplot):
                                    if field_labels[ilabel]=="uwnd":
                                        varmin=subset.uminmax[0]
                                        varmax=subset.uminmax[1]
                                    elif field_labels[ilabel]=="wwnd":
                                        varmin=subset.wminmax[0]
                                        varmax=subset.wminmax[1]
                                    elif field_labels[ilabel]=="dens":
                                        varmin=subset.densminmax[0]
                                        varmax=subset.densminmax[1]
                                    else:  
                                        assert field_labels[ilabel]=="potentialtemperature"
                                        varmin=subset.ptminmax[0]
                                        varmax=subset.ptminmax[1]
                                    #print(varmin, varmax)
                                    tarplot[:, ivar, :, :] = tar[:, ivar, :, :]*(varmax-varmin) + varmin
                                    outputplot[:, ivar, :, :]=output_leadtime[:, ivar, :, :]*(varmax-varmin) + varmin
                                    output_autoplot[:, ivar, :, :]=output_auto[:, ivar, :, :]*(varmax-varmin) + varmin
                                filename = os.path.join(self.output_dir,f"./{setname}_{isample.item()}_{self.model.embed_ensemble[-1].patch_size[0]}_LT{int(leadtime[0,0].item())}.png")
                                self.plot_visual_contourcompthree(tarplot, outputplot, output_autoplot, [field_labels[ilabel] for ilabel in labelsplot],filename,
                                                                lt=int(leadtime[0,0].item()), x_inp=inp)
                                
                                #plotting potentialtemperature for demonstration
                                if False:
                                    varmin=subset.ptminmax[0]
                                    varmax=subset.ptminmax[1]
                                    iplot = [field_labels[ilabel] for ilabel in labelsplot].index("potentialtemperature")
                                    plot_var = tar[0, iplot, :, :]*(varmax-varmin) + varmin
                                    self.plot_visual_contourindivi(plot_var, os.path.join(self.output_dir,
                                                                                        f"./PT_tar_{setname}_{isample.item()}_{self.model.embed_ensemble[-1].patch_size[0]}_LT{int(leadtime[0,0].item())}.png"))
                                    plot_var = output_leadtime[0, iplot, :, :]*(varmax-varmin) + varmin
                                    self.plot_visual_contourindivi(plot_var, os.path.join(self.output_dir,
                                                                                    f"./PT_pred_{setname}_{isample.item()}_{self.model.embed_ensemble[-1].patch_size[0]}_LT{int(leadtime[0,0].item())}.png"))  
                self.single_print('DONE INFERENCE %s'%setname)
            sys.exit(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/media/6pz/DATA/projects/multiscale_multiphysics/cades/token_red/runs_adan_miniweather_frontier/tokens_32_ltime20/val_logs/', type=str)
    #parser.add_argument("--model_dir", default='/media/6pz/DATA/projects/multiscale_multiphysics/cades/token_red/runs_adan_miniweather_frontier/tokens_32/val_logs/', type=str)
    parser.add_argument("--yaml_config", default='hyperparams.yaml', type=str)
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--data_dir", default="/media/6pz/DATA/projects/CFD-datasets/miniweather/")
    args = parser.parse_args()
    params = YParams(os.path.join(args.model_dir, args.yaml_config))
    params.use_ddp = args.use_ddp
    if hasattr(params,'train_data_paths'):
        for iset in range(len(params.train_data_paths)):
            params.train_data_paths[iset][0]=args.data_dir
    if hasattr(params,'valid_data_paths'):
        for iset in range(len(params.valid_data_paths)):
            params.valid_data_paths[iset][0]=args.data_dir
    if hasattr(params,'test_data_paths'):
        for iset in range(len(params.test_data_paths)):
            params.test_data_paths[iset][0]=args.data_dir
    else:
        params.test_data_paths = params.valid_data_paths
    if hasattr(params,'leadtime_max') and params.leadtime_max>1:
        params.train_leadtime_max = params.leadtime_max
        params.leadtime_max=100
        print("leadtime range [1, %d] in training; for inference, we expand it to [1, %d]"%
              (params.train_leadtime_max, params.leadtime_max))
    
    print(params)    
    # Set up distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.use_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank) # Torch docs recommend just using device, but I had weird memory issues without setting this.
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")
   
    params['experiment_dir'] = args.model_dir
    params['checkpoint_path'] = os.path.join(args.model_dir, "best_ckpt.tar")
    #params.output = os.path.join(args.model_dir,"./autoregressive/")
    params.output = os.path.join(args.model_dir,"./plots_output/")
    if not os.path.exists(params.output):
        os.makedirs(params.output)

    # Have rank 0 check for and/or make directory
    if  global_rank==0:
        if not os.path.isdir(params['experiment_dir'] ):
            raise ValueError("Cannot find path %s"%params['experiment_dir'])
    if os.path.isfile(params.checkpoint_path):
        params['resuming'] = True 
    else:
        sys.exit("checkpoint_path not found: %s" % params.checkpoint_path)

    torch.backends.cudnn.benchmark = False

    inferencer = Inferencer(params, global_rank, local_rank, device)
    inferencer.inference()
    inferencer.inference_extra(LT=40)
    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
