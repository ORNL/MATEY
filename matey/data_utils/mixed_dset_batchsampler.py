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
                 global_rank=0, group_size=1, distributed=True, num_sp_groups=None):
        self.batch_size_base = batch_size
        self.sub_dsets = dataset.sub_dsets
        self.ordered_sampling=ordered_sampling
        if distributed:
            """
            supporting varying batch sizes across dataset & multiple ranks coordinate together to load the same sample

            For world_size ranks, split them into "group_size" X "num_sp_groups" 2D ranks;
            For every group with group_size ranks, they read the subparts from the same sample, seeded by group_id
            All num_sp_groups read from the same dataset, seeded by a constant 0 (FIXME: this can/should be related to multiple datasets later)
            So the actual batch size is: "self.batch_size" X "num_sp_groups"

            when "ordered_sampling" is True: sampling across subset squentially following a deterministic order for every "nbatchs_loc" batchs
            """
            self.sub_samplers = []
            self.batch_size = []
            for subset in self.sub_dsets:
                if subset.type in dataset.DP_dsets:
                    batch_size_subset = self._determine_batchsize_(subset)
                    group_id=global_rank//group_size #rank of current group within num_sp_groups
                    num_replicas=num_sp_groups
                    dset_rank=0 #all num_sp_groups groups read from the same datset
                    ##dset_rank=group_id #allow each group read from different dataset: not work as different model parts (FIXME)
                else:
                    batch_size_subset = self.batch_size_base
                    group_id=global_rank
                    num_replicas=None
                    dset_rank=0
                self.batch_size.append(batch_size_subset)
                self.sub_samplers.append(base_sampler(subset, drop_last=drop_last, num_replicas=num_replicas, rank=group_id, shuffle=shuffle))
        else:
            self.sub_samplers = [base_sampler(subset) for subset in self.sub_dsets]
            self.batch_size = [batch_size for _ in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.max_samples = max_samples
        self.global_rank = global_rank
        self.group_size = group_size
        self.rank = dset_rank
        self.batches_perset = [len(sampler)//batchsize for sampler, batchsize in zip(self.sub_samplers, self.batch_size)]
        #acutal total minibatches
        self.len_batchsamplers = sum(self.batches_perset)
        self.iset_choices = torch.tensor([iset for iset, n in enumerate(self.batches_perset) for _ in range(n)], dtype=torch.long)
        min_batches = min(self.batches_perset)
        self.iset_choices_ordered_truc = []
        for _ in range(min_batches//nbatchs_loc):
            for iset in range(len(self.batches_perset)):
                for _ in range(nbatchs_loc):
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
            choices_t = self.iset_choices_ordered_truc[:self.max_samples]
        else:
            generator = torch.Generator().manual_seed(5000*self.epoch+100*self.seed+self.rank)
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

