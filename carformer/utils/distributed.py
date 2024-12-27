import torch

# DistributedDataParallel required imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os  # For os.environ


def disable_printing(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# For distributed training, we need to use the following function
# This function sets up the environment for distributed training, and lets every process know
# which process it is (rank) and how many processes there are (world_size)
def ddp_setup(args, dist_url="env://"):
    # We get the rank and world size from the environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if args.gpus != world_size:
            # Warn
            print(
                "WARNING: args.gpus != os.environ['WORLD_SIZE']",
                flush=True,
            )
        args.gpus = int(os.environ["WORLD_SIZE"])

    print(
        "| Initializing process with rank {} out of {} processes |".format(
            rank, world_size
        ),
        flush=True,
    )

    # This is a useful hack I like to do sometimes. This DISABLES printing on nodes that are not rank 0, to make the output cleaner
    disable_printing(rank == 0)
    torch.cuda.set_device(rank)

    init_process_group(
        # backend="nccl", # just GPU, commented out in order to use both CPU and GPU
        init_method=dist_url,
        rank=rank,
        world_size=args.gpus,
    )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
