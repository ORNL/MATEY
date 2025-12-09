import os
import re
from functools import reduce
from operator import mul
import torch
import torch.distributed as dist
from einops import rearrange
from datetime import timedelta
from torch.optim.lr_scheduler import CosineAnnealingLR

def check_sp(sequence_parallel_groups, global_rank):
    for groupid, group in enumerate(sequence_parallel_groups):
        try:
            group_rank = dist.get_group_rank(group, global_rank)
            if group_rank != -1:  
                current_group = group
                print(f"Rank {global_rank} is in group {groupid}, {group} with local rank {group_rank}")
        except ValueError as e:
            pass
        
def setup_dist(params):
    #num_gpus_per_node = torch.cuda.device_count()
    world_size = int(os.environ['SLURM_NTASKS'])
    global_rank = rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    # os.environ['MASTER_ADDR'] = str(args.master_addr)
    # os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'
    if os.getenv("SLURM_STEP_NODELIST") is not None:
        os.environ['MASTER_ADDR']  = parse_slurm_nodelist(os.environ["SLURM_STEP_NODELIST"])[0]

    if params.use_ddp or params.use_fsdp:
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)

    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return device, world_size, local_rank, global_rank

def closest_factors(n, dim):
    #temporary, from Andrey's
    assert n > 0 and dim > 0, f"{n} and {dim} must be greater than 0"

    if dim == 1:
        return [n]

    factors = []
    i = 2
    nn = n
    while nn > 1:
        while nn % i == 0:
            factors.append(i)
            nn //= i
        i += 1

    # Reduce the list of factors to match the dimension (dim)
    while len(factors) > dim:
        # Combine the two smallest factors
        factors[1] *= factors[0]
        factors.pop(0)
        factors.sort()

    assert reduce(mul, factors) == n and len(factors)==dim

    return factors

def get_rank_ingroup(rank, group):
    try:
        group_rank = dist.get_group_rank(group, rank)
        return group_rank
    except:
        return -1

def get_sequence_parallel_group(sequence_parallel_groupsize=None, num_sequence_parallel_groups=None):
    """
    Create sequence parallel groups based on number of sequence_parallel_groups.
    """
    world_size = dist.get_world_size()
    if sequence_parallel_groupsize is None:
        sequence_parallel_size=world_size//num_sequence_parallel_groups
    else:
        sequence_parallel_size = sequence_parallel_groupsize
    sequence_parallel_groups = []
    for start in range(0, world_size, sequence_parallel_size):
        ranks = list(range(start, start + sequence_parallel_size))
        sequence_parallel_group = dist.new_group(ranks,timeout=timedelta(minutes=40))
        sequence_parallel_groups.append(sequence_parallel_group)
    return sequence_parallel_groups, sequence_parallel_size

def locate_group(sequence_parallel_groups, global_rank):
    if len(sequence_parallel_groups)==dist.get_world_size():
        return None, global_rank
    for group in sequence_parallel_groups:
        group_rank = get_rank_ingroup(global_rank, group)
        if group_rank>-1:
            return group, group_rank
    raise ValueError(f"global_rank {group_rank} is not in [0, {sequence_parallel_groups})")

def splitsample(x, y, sequence_parallel_groups, blockdict=None):
    """
    split a sample based on sequence split groups
    """
    D, H, W = x.shape[3:]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_sequence_parallel_groups = len(sequence_parallel_groups)
    sequence_parallel_size=world_size//num_sequence_parallel_groups
    ##############################################################
    #based on sequence_parallel_size, split the data in D, H, W direciton
    nproc_blocks = closest_factors(sequence_parallel_size, 3)
    assert reduce(mul, nproc_blocks)==sequence_parallel_size
    ##############################################################
    #split a sample by space into nprocz blocks for z-dim, nprocx blocks for x-dim, and nprocy blocks for y-dim
    Dloc = D//nproc_blocks[0]
    Hloc = H//nproc_blocks[1]
    Wloc = W//nproc_blocks[2]
    #B,T,C,D,H,W-->B,T,C,(sequence_parallel_size=nprocz*nprocx*nprocy),psz,psx,psy
    xsplits = x.unfold(3, Dloc, Dloc).unfold(4, Hloc, Hloc).unfold(5, Wloc, Wloc).flatten(start_dim=3, end_dim=5)
    #B,C,D,H,W-->B,C,(sequence_parallel_size),psz,psx,psy
    ysplits = y.unfold(2, Dloc, Dloc).unfold(3, Hloc, Hloc).unfold(4, Wloc, Wloc).flatten(start_dim=2, end_dim=4)
    xsplits=rearrange(xsplits,'b t c npb d h w -> npb b t c d h w').contiguous()
    ysplits=rearrange(ysplits,'b c npb d h w -> npb b c d h w').contiguous()
    ##############################################################
    #keep track of each block/split ID
    iz, ix, iy = torch.meshgrid(torch.arange(nproc_blocks[0]), 
                                torch.arange(nproc_blocks[1]),  
                                torch.arange(nproc_blocks[2]), indexing="ij")
    blockIDs = torch.stack([iz.flatten(), ix.flatten(), iy.flatten()], dim=-1) #[sequence_parallel_size, 3]
    if blockdict is not None:
        Lz, Lx, Ly = blockdict["Lzxy"]
        Lz_start, Lx_start, Ly_start = blockdict["zxy_start"]
        d_dim, h_dim, w_dim = blockdict["Ind_dim"] 
    else:
        blockdict={}
        Lz, Lx, Ly = 1.0, 1.0, 1.0
        Lz_start, Lx_start, Ly_start = 0.0, 0.0, 0.0
        d_dim, h_dim, w_dim = D, H, W
    
    blockdict["Lzxy"] = [Lz/nproc_blocks[0], Lx/nproc_blocks[1], Ly/nproc_blocks[2]]
    blockdict["nproc_blocks"] = nproc_blocks
    blockdict["Ind_dim"] = [d_dim//nproc_blocks[0], h_dim//nproc_blocks[1], w_dim//nproc_blocks[2]]
    #######################
    #get the split at group_rank of x and y
    for group in sequence_parallel_groups:
        group_rank = get_rank_ingroup(rank, group)
        if 0<=group_rank<sequence_parallel_size:
            #correct position
            idz, idx, idy = blockIDs[group_rank,:]
            Lz_loc, Lx_loc, Ly_loc = blockdict["Lzxy"]
            blockdict["zxy_start"] = [Lz_start+idz*Lz_loc, Lx_start+idx*Lx_loc, Ly_start+idy*Ly_loc]
            return xsplits[group_rank,...], ysplits[group_rank,...], blockdict
        else:
            continue
    raise ValueError(f"unkown rank {rank}, not taken by any group")

def splitsample_rank0(x, y, sequence_parallel_groups, blockdict=None):
    """
    split a sample based on sequence split groups, operated only on group_rank=0
    """
    D, H, W = x.shape[3:]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_sequence_parallel_groups = len(sequence_parallel_groups)
    sequence_parallel_size=world_size//num_sequence_parallel_groups
    ##############################################################
    #based on sequence_parallel_size, split the data in D, H, W direciton
    nproc_blocks = closest_factors(sequence_parallel_size, 3)
    assert reduce(mul, nproc_blocks)==sequence_parallel_size
    ##############################################################
    #split a sample by space into nprocz blocks for z-dim, nprocx blocks for x-dim, and nprocy blocks for y-dim
    Dloc = D//nproc_blocks[0]
    Hloc = H//nproc_blocks[1]
    Wloc = W//nproc_blocks[2]
    #B,T,C,D,H,W-->B,T,C,(sequence_parallel_size=nprocz*nprocx*nprocy),psz,psx,psy
    xsplits = x.unfold(3, Dloc, Dloc).unfold(4, Hloc, Hloc).unfold(5, Wloc, Wloc).flatten(start_dim=3, end_dim=5)
    #B,C,D,H,W-->B,C,(sequence_parallel_size),psz,psx,psy
    ysplits = y.unfold(2, Dloc, Dloc).unfold(3, Hloc, Hloc).unfold(4, Wloc, Wloc).flatten(start_dim=2, end_dim=4)
    xsplits=rearrange(xsplits,'b t c npb d h w -> npb b t c d h w').contiguous()
    ysplits=rearrange(ysplits,'b c npb d h w -> npb b c d h w').contiguous()
    ##############################################################
    #keep track of each block/split ID
    iz, ix, iy = torch.meshgrid(torch.arange(nproc_blocks[0]), 
                                torch.arange(nproc_blocks[1]),  
                                torch.arange(nproc_blocks[2]), indexing="ij")
    blockIDs = torch.stack([iz.flatten(), ix.flatten(), iy.flatten()], dim=-1) #[sequence_parallel_size, 3]
    if blockdict is not None:
        Lz, Lx, Ly = blockdict["Lzxy"]
        Lz_start, Lx_start, Ly_start = blockdict["zxy_start"]
        d_dim, h_dim, w_dim = blockdict["Ind_dim"] 
    else:
        blockdict={}
        Lz, Lx, Ly = 1.0, 1.0, 1.0
        Lz_start, Lx_start, Ly_start = 0.0, 0.0, 0.0
        d_dim, h_dim, w_dim = D, H, W
    
    blockdict["Lzxy"] = [Lz/nproc_blocks[0], Lx/nproc_blocks[1], Ly/nproc_blocks[2]]
    blockdict["nproc_blocks"] = nproc_blocks
    blockdict["Ind_dim"] = [d_dim/nproc_blocks[0], h_dim/nproc_blocks[1], w_dim/nproc_blocks[2]]
    #######################
    blockdict["zxy_start"]=[]
    for group_rank in range(sequence_parallel_size):
        #correct position
        idz, idx, idy = blockIDs[group_rank,:]
        Lz_loc, Lx_loc, Ly_loc = blockdict["Lzxy"]
        blockdict["zxy_start"].append([Lz_start+idz*Lz_loc, Lx_start+idx*Lx_loc, Ly_start+idy*Ly_loc])
    #[npb b t c d h w] for xplits; [npb b c d h w] for ysplits; blockdict["zxy_start"] contains list of "zxy_start" for all members/ranks inside group
    return xsplits, ysplits, blockdict

def parse_slurm_nodelist(nodelist):
    #from: https://github.com/ORNL/HydraGNN/blob/40524b2ebc2e7b9c61aa2280af7d64f3d69d3f85/hydragnn/utils/distributed.py#L53
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)

    return nlist

def add_weight_decay(model, weight_decay=1e-5, inner_lr=1e-3, skip_list=()):
    """ From Ross Wightman at:
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

        Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale)
        then don't apply weight decay.
        """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (len(param.squeeze().shape) <= 1 or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
            {'params': no_decay, 'weight_decay': 0.,},
            {'params': decay, 'weight_decay': weight_decay}]
class CosineNoIncrease(CosineAnnealingLR):
    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min] * len(self.base_lrs)
        return super().get_lr()

