import torch
import os


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend='nccl', init_method="tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']),
                                        world_size=world_size, rank=local_rank)

    torch.distributed.barrier()

    print(os.environ["LOCAL_RANK"])

    tensor = torch.randn(2, 4).to(local_rank)
    print(f"{local_rank} tensor: {tensor}")
    tensors_gather = [ torch.zeros_like(tensor) for i in range(world_size) ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    print(f"{local_rank} gathered:{tensors_gather}")