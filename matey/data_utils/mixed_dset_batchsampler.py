# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

from typing import  Iterator
import torch
from torch.utils.data import Sampler, BatchSampler, Dataset
import functools, operator, math

class MultisetBatchSampler(BatchSampler):
    r"""Batch Sampler that samples from multiple datasets with samples inside each mini-batch from a specific dataset.
    """
    def __init__(self, dataset: Dataset, base_sampler:Sampler, batch_size: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True, max_samples=10, ordered_sampling=True, nbatchs_loc=5,
                 distributed=True):
        self.batch_size_base = batch_size
        self.sub_dsets = dataset.sub_dsets
        self.ordered_sampling=ordered_sampling
        self.mixed_dset_opt=dataset.mixed_dset_opt
        if self.mixed_dset_opt and not self.ordered_sampling:
            raise ValueError("mixed_dset_opt is only supported for now in ordered_sampling; will revisit as needed for other cases")        
        #dsets group by tkhead {groupID: [iset ...]}
        self.dsets_groupbytk = dataset.dsets_groupbytk
        self.setgroup2rankgroup=dataset.setgroup2rankgroup
        if distributed:
            """
            supporting varying batch sizes across dataset & multiple ranks coordinate together to load the same sample

            For world_size ranks, split them into "group_size" X "num_sp_groups" 2D ranks;
            For every group with group_size ranks, they read the subparts from the same sample, seeded by group_id
            if self.mixed_dset_opt is False:
                All num_sp_groups read from the same dataset, seeded by a constant 0 
                So the actual batch size is: "self.batch_size" X "num_sp_groups"
            else:
                allow different SP groups to sample from a mixture of datasets in two steps
                1. all SP groups randomly sample a datasetgroup id with the same seed --> so all the same datasetgroup
                2. From the datasetgroup id, each SP group randomly pick a dataset in the same group from 
                    self.setgroup2rankgroup[datasetgroup id], seeded with group_id --> all ranks in the same group read the same sample 

            when "ordered_sampling" is True: sampling across subset squentially following a deterministic order for every "nbatchs_loc" batchs
            """
            self.sub_samplers = []
            self.batch_size = []
            for iset, subset in enumerate(self.sub_dsets):
                if subset.type in dataset.DP_dsets:
                    batch_size_subset = self._determine_batchsize_(subset)
                else:
                    batch_size_subset = self.batch_size_base
                sampler_rank = dataset.dsets_spconfig[iset]["sampler_rank"]
                sampler_num_replicas = dataset.dsets_spconfig[iset]["sampler_num_replicas"]
                self.batch_size.append(batch_size_subset)
                self.sub_samplers.append(base_sampler(subset, drop_last=drop_last, num_replicas=sampler_num_replicas, rank=sampler_rank, shuffle=shuffle))
        else:
            self.sub_samplers = [base_sampler(subset) for subset in self.sub_dsets]
            self.batch_size = [batch_size for _ in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.max_samples = max_samples
        #dset_seed when constant (0) across groups, all num_sp_groups groups read from the same datset
        #when different, allow each group read from different dataset, but all ranks from the same group must read the same data sample
        self.dset_seed = self.setgroup2rankgroup if self.mixed_dset_opt else 0
        self.batches_perset = [len(sampler)//batchsize for sampler, batchsize in zip(self.sub_samplers, self.batch_size)]
        #acutal total minibatches
        self.len_batchsamplers = sum(self.batches_perset)
        self.iset_choices = torch.tensor([iset for iset, n in enumerate(self.batches_perset) for _ in range(n)], dtype=torch.long)
        min_batches = min(self.batches_perset)
        self.nbatchs_loc=nbatchs_loc
        if min_batches<self.nbatchs_loc:
            self.nbatchs_loc = min_batches
        self.iset_choices_ordered_truc = []
        for _ in range(min_batches//self.nbatchs_loc):
            for iset in range(len(self.batches_perset)):
                for _ in range(self.nbatchs_loc):
                    self.iset_choices_ordered_truc.append(iset)
        self.iset_choices_ordered_truc = torch.tensor(self.iset_choices_ordered_truc, dtype=torch.long)
        if not self.ordered_sampling and len(self.iset_choices)<self.max_samples:
            print(f"Warning: asked for max_samples {self.max_samples}, but only have {len(self.iset_choices)}, {dataset.path_list}")
            self.max_samples=len(self.iset_choices)
        if self.ordered_sampling and len(self.iset_choices_ordered_truc)<self.max_samples:
            print(f"Warning: asked for max_samples {self.max_samples}, but only have {len(self.iset_choices_ordered_truc)}, {dataset.path_list}")
            self.max_samples=len(self.iset_choices_ordered_truc)

    def _determine_batchsize_(self, subset, threelevels=False, refer_datasize=[256, 256, 256, 4]):
        #FIXME: currently heuristic, should be improved based on performance
        if not threelevels:
            probsize = functools.reduce(operator.mul, subset.cubsizes)*len(subset.field_names)
            probsize_ref = functools.reduce(operator.mul, refer_datasize)
            ratio = probsize_ref//probsize
            if ratio>0:
                expo = ratio.bit_length() - 1
                return int(min(self.batch_size_base*2**expo, 16))
            else:
                ratio = math.ceil(probsize/probsize_ref)
                expo = ratio.bit_length() - 1
                return int(max(self.batch_size_base/2**expo, 1))
        else:
            if subset.type in ["MHD256"]:
                return self.batch_size_base//2
            elif subset.type in ['swe','incompNS','diffre2d', 'compNS','compNS128','compNS512', 'thermalcollision2d',
                                'planetswe', 'euleropen', 'eulerperiodic','rayleighbenard', 'shearflow', 'turbradlayer2D', 'viscoelastic']:
                return self.batch_size_base*4
            else:
                return self.batch_size_base

    def __iter__(self):
        #batch sampler
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        if self.ordered_sampling:
            base = self.iset_choices_ordered_truc.view(-1, len(self.batches_perset), self.nbatchs_loc)
            out = []
            if self.mixed_dset_opt:
                #same group id sampler for all gpus so that they always sample from datasets in same group
                generator_group = torch.Generator().manual_seed(5000*self.epoch + 100*self.seed) 
                groupids = list(self.dsets_groupbytk.keys())    
                generator = [torch.Generator().manual_seed(5000*self.epoch + 100*self.seed + self.dset_seed[groupid]) for groupid in groupids]
                for cycle in base:
                    out_oneround=[]
                    permgp = torch.randperm(len(groupids), generator=generator_group)
                    groupids_perm = [groupids[i] for i in permgp]
                    for igroup in groupids_perm:
                        subsets=self.dsets_groupbytk[igroup]
                        #perm order of datasets with nbatchs_loc as a unit
                        perm = torch.randperm(len(subsets), generator=generator[igroup])
                        out_oneround.append(cycle[subsets[perm]])
                    out_oneround = torch.cat(out_oneround, dim=0)
                    out.append(out_oneround)
            else:
                generator = torch.Generator().manual_seed(5000*self.epoch + 100*self.seed + self.dset_seed)
                for cycle in base:
                    #perm order of datasets with nbatchs_loc as a unit
                    perm = torch.randperm(cycle.shape[0], generator=generator)
                    out.append(cycle[perm])
            choices_t = torch.cat(out, dim=0).flatten()[:self.max_samples]
            #choices_t = self.iset_choices_ordered_truc[:self.max_samples]
        else:
            generator = torch.Generator().manual_seed(5000*self.epoch+100*self.seed+self.dset_seed)
            perm      = torch.randperm(len(self.iset_choices), generator=generator)
            choices_t = self.iset_choices[perm][:self.max_samples]
        
        offsets = [max(0, off) for off in self.dataset.offsets]
        
        for subset_idx in choices_t:
            idx = subset_idx.item()
            it, off = samplers[idx], offsets[idx]
            yield [next(it) + off for _ in range(self.batch_size[idx])]
        
    def __len__(self) -> int:
        return self.max_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)
        self.epoch = epoch

