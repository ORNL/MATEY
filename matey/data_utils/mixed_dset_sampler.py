from typing import  Iterator
import torch
from torch.utils.data import Sampler, Dataset
import functools, operator, math

class MultisetSampler(Sampler):
    r"""Sampler that samples from multiple datasets with samples inside each mini-batch from a specific dataset.
    """
    def __init__(self, dataset: Dataset, base_sampler:Sampler, batch_size: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True, max_samples=10,
                 global_rank=0, group_size=1, distributed=True, num_sp_groups=None) -> None:
        #self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        if distributed:
            #self.sub_samplers = [base_sampler(dataset, drop_last=drop_last, num_replicas=num_replicas, rank=rank, shuffle=shuffle) 
            #                     for dataset in self.sub_dsets]
            """
            For world_size ranks, split them into group_size X num_sp_groups 2D ranks; 
            For every group with group_size ranks, they read the subparts from the same sample, seeded by gound_id
            All num_sp_groups read from the same dataset, seeded by a constant 0 (is this necessary? Pei)
            So the actual batch size is: self.batch_size X num_sp_groups
            """

            self.sub_samplers = []
            self.batch_size = []
            for subset in self.sub_dsets:
                if False:
                    if subset.type in ["MHD256"]:
                        self.batch_size.append(batch_size//2)
                    elif subset.type in ['swe','incompNS','diffre2d', 'compNS','compNS128','compNS512', 'thermalcollision2d', 
                                        'planetswe', 'euleropen', 'eulerperiodic','rayleighbenard', 'shearflow', 'turbradlayer2D', 'viscoelastic']:
                        self.batch_size.append(batch_size*4)
                    else:
                        self.batch_size.append(batch_size)
                else:
                    probsize = functools.reduce(operator.mul,subset.cubsizes)*len(subset.field_names)
                    ratio = (256*256*256*4)//probsize #FIXME: hard-coded for now; using hit as a reference
                    if ratio>0:
                        expo = ratio.bit_length() - 1
                        self.batch_size.append(int(min(batch_size*2**expo, 16)))
                    else:
                        ratio = math.ceil(probsize/(256*256*256*4))
                        expo = ratio.bit_length() - 1
                        self.batch_size.append(int(max(batch_size/2**expo, 1)))
                print(f"Pei debugging, {subset.type}, self.batch_size, {self.batch_size}, {subset.cubsizes}, {len(subset.field_names)}", flush=True)


                if subset.type in dataset.DP_dsets:
                    group_id=global_rank//group_size #rank of current group within num_sp_groups
                    num_replicas=num_sp_groups
                    dset_rank=0 #all num_sp_groups groups read from the same datset
                    ##dset_rank=group_id #allow each group read from different dataset: not work as different model parts
                else:
                    group_id=global_rank
                    num_replicas=None
                    dset_rank=0
                self.sub_samplers.append(base_sampler(subset, drop_last=drop_last, num_replicas=num_replicas, rank=group_id, shuffle=shuffle)) 
        else:
            self.sub_samplers = [base_sampler(subset) for subset in self.sub_dsets]
            self.batch_size = [batch_size for _ in self.sub_dsets]
        self.len_samplers = sum([len(sampler)//batchsize for sampler, batchsize in zip(self.sub_samplers, self.batch_size)]) 
        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.max_samples = max_samples
        self.global_rank = global_rank
        self.group_size = group_size
        self.rank = dset_rank
        self.batches_perset = [len(sampler)//batchsize for sampler, batchsize in zip(self.sub_samplers, self.batch_size)]
        self.iset_choices = torch.tensor([iset for iset, n in enumerate(self.batches_perset) for _ in range(n)], dtype=torch.long)
        min_batches = min(self.batches_perset)
        #self.iset_choices_truc = torch.tensor([iset for _ in range(min_batches) for iset in range(len(self.batches_perset)) ], dtype=torch.long)
        self.iset_choices_ordered_truc = []
        for _ in range(min_batches//5): 
            for iset in range(len(self.batches_perset)):
                for _ in range(5):
                    self.iset_choices_ordered_truc.append(iset)
        
        self.iset_choices_ordered_truc = torch.tensor(self.iset_choices_ordered_truc, dtype=torch.long)

        if len(self.iset_choices)<self.max_samples:
            print(f"Warning: asked for max_samples {self.max_samples}, but only have {len(self.iset_choices)}, {dataset.path_list}")
            self.max_samples=len(self.iset_choices)

    def __iter__(self):
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        generator = torch.Generator().manual_seed(5000*self.epoch+100*self.seed+self.rank)
        perm      = torch.randperm(len(self.iset_choices), generator=generator)
        choices_t = self.iset_choices[perm][:self.max_samples]
        offsets = [max(0, off) for off in self.dataset.offsets]
        #print(f"Pei debugging, {self.rank}, {self.global_rank}, {self.group_size}, {choices_t[:10]}, {perm}", flush=True)
        
        #for subset_idx in choices_t:
        for subset_idx in self.iset_choices_ordered_truc:
            idx = subset_idx.item()
        #for idx in range(len(self.sub_samplers)):
            #print("debugging", len(samplers), len(offsets), subset_idx, samplers, offsets)
            it, off = samplers[idx], offsets[idx]
            #batch sampler
            yield [next(it) + off for _ in range(self.batch_size[idx])]

            #for _ in range(self.batch_size[idx]):
            #    yield next(it) + off
        """
        sampler_choices = list(range(len(samplers)))
        count = 0
        while len(sampler_choices) > 0:
            # count += 1 # old location of count update, leads to missed batches
            index_sampled = torch.randint(0, len(sampler_choices), size=(1,), generator=generator).item()
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            # Do drop last batch type logic - if you can get a full batch, yield it, otherwise move to next dataset
            try:
                queue = [next(samplers[dset_sampled]) + offset for _ in range(self.batch_size)]
                #if len(queue) == self.batch_size:
                count += 1  # new location of count update, only update if successful
                for d in queue:
                    yield d
            except Exception as err:
                # print('ERRRR', err)
                # sampler_choices.pop(index_sampled)
                # print(f'Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}')
                # continue
                sampler_choices.pop(index_sampled)
                if self.rank ==0:
                    print(f'Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}', flush= True)
                continue
            if count >= self.max_samples:
                break
        """
    def __len__(self) -> int:
        return self.len_samplers

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

