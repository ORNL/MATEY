from typing import  Iterator
import torch
from torch.utils.data import Sampler, Dataset

class MultisetSampler(Sampler):
    r"""Sampler that samples from multiple datasets with samples inside each mini-batch from a specific dataset.
    """
    def __init__(self, dataset: Dataset, base_sampler:Sampler, batch_size: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True, max_samples=10,
                 rank=0, distributed=True, num_replicas=None) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        if distributed:
            self.sub_samplers = [base_sampler(dataset, drop_last=drop_last, num_replicas=num_replicas, rank=rank, shuffle=shuffle) 
                                 for dataset in self.sub_dsets]
        else:
            self.sub_samplers = [base_sampler(dataset) for dataset in self.sub_dsets]
        self.len_samplers = sum([len(sampler) for sampler in self.sub_samplers]) 
        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank
        self.batches_perset = [len(sampler)//self.batch_size for sampler in self.sub_samplers]
        self.iset_choices = torch.tensor([iset for iset, n in enumerate(self.batches_perset) for _ in range(n)], dtype=torch.long)
        if len(self.iset_choices)<self.max_samples:
            print(f"Warning: asked for max_samples {self.max_samples}, but only have {len(self.iset_choices)}, {dataset.path_list}")
            self.max_samples=len(self.iset_choices)

    def __iter__(self):
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        generator = torch.Generator().manual_seed(5000*self.epoch+100*self.seed+self.rank)
        perm      = torch.randperm(len(self.iset_choices), generator=generator)
        choices_t = self.iset_choices[perm][:self.max_samples]
        offsets = [max(0, off) for off in self.dataset.offsets]
        
        for subset_idx in choices_t:
            idx = subset_idx.item()
            it, off = samplers[idx], offsets[idx]
            for _ in range(self.batch_size):
                yield next(it) + off
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

