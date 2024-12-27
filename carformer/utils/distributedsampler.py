import torch
import torch.distributed as dist
import math
import numpy as np
import warnings


class WeightedDistributedSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        bins: a list of bins containing the indices of the dataset in each bin. Bins are chosen at equal probability, and then a sample is chosen from the bin at equal probability.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(
        self,
        dataset,
        subsample_ratio=1.0,
        num_replicas=None,
        rank=None,
        weights=None,
        shuffle=False,
    ):
        # If weights, make sure they are a torch tensor
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float)
        if num_replicas is None:
            if not dist.is_available():
                # Default to single process
                num_replicas = 1
            else:
                if dist.is_initialized():
                    num_replicas = dist.get_world_size()
                else:
                    # Warn user that num_replicas is set to 1
                    warnings.warn(
                        "DistributedSampler is initialized without "
                        "dist being initialized. Set num_replicas "
                        "to 1 instead."
                    )
                    num_replicas = 1
        if rank is None:
            if not dist.is_available():
                # Default to single process
                rank = 0
            else:
                if dist.is_initialized():
                    rank = dist.get_rank()
                else:
                    # Warn user that rank is set to 0
                    warnings.warn(
                        "DistributedSampler is initialized without "
                        "dist being initialized. Set rank "
                        "to 0 instead."
                    )
                    rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas * subsample_ratio)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.weights = weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.weights is not None:
            if self.epoch == 0:
                print("Generating indices with weights for epoch {}".format(self.epoch))
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=g
            ).tolist()
            # Print histogram of indices
            if self.epoch == 0:
                hist = np.histogram(
                    np.asarray(indices), bins=range(0, len(self.dataset) + 1)
                )
                print(hist)
        else:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        # print(self.rank, indices[:100])
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_subsample_ratio(self, subsample_ratio):
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas * subsample_ratio)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# # Bin sampler
# class BinDistributedSampler(torch.utils.data.Sampler):
#     """Sampler that restricts data loading to a subset of the dataset.
#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSampler instance as a DataLoader sampler,
#     and load a subset of the original dataset that is exclusive to it.
#     .. note::
#         Dataset is assumed to be of constant size.
#     Arguments:
#         dataset: Dataset used for sampling.
#         num_replicas (optional): Number of processes participating in
#             distributed training.
#         rank (optional): Rank of the current process within num_replicas.
#         shuffle (optional): If true (default), sampler will shuffle the indices
#     """

#     def __init__(
#         self,
#         dataset,
#         subsample_ratio=1.0,
#         num_replicas=None,
#         rank=None,
#         shuffle=False,
#     ):
#         if num_replicas is None:
#             if not dist.is_available():
#                 # Default to single process
#                 num_replicas = 1
#             else:
#                 if dist.is_initialized():
#                     num_replicas = dist.get_world_size()
#                 else:
#                     # Warn user that num_replicas is set to 1
#                     warnings.warn(
#                         "DistributedSampler is initialized without "
#                         "dist being initialized. Set num_replicas "
#                         "to 1 instead."
#                     )
#                     num_replicas = 1
#         if rank is None:
#             if not dist.is_available():
#                 # Default to single process
#                 rank = 0
#             else:
#                 if dist.is_initialized():
#                     rank = dist.get_rank()
#                 else:
#                     # Warn user that rank is set to 0
#                     warnings.warn(
#                         "DistributedSampler is initialized without "
#                         "dist being initialized. Set rank "
#                         "to 0 instead."
#                     )
#                     rank = 0

#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.num_samples = int(
#             math.ceil(len(self.dataset) * 1.0 / self.num_replicas * subsample_ratio)
#         )
#         self.total_size = self.num_samples * self.num_replicas
#         self.shuffle = shuffle

#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self
