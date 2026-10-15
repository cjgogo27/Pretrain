"""Two-process local Gloo launcher, including Windows builds without libuv.

Usage: python examples/launch_cpu.py [train.py arguments]
Calls the real train.run; this is not an elastic/multi-machine launcher.
"""
from datetime import timedelta
import os
import socket
import sys

import torch.distributed as dist
import torch.multiprocessing as mp

import train


def worker(rank, port, arguments):
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    sys.argv = ["train.py", *arguments]
    args = train.parse_args()
    if args.device != "cpu":
        raise ValueError("launch_cpu supports only --device cpu")
    try:
        store = dist.TCPStore("127.0.0.1", port, 2, rank == 0,
                              timeout=timedelta(seconds=90), use_libuv=False)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                                timeout=timedelta(seconds=90))
        train.run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(arguments):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # A concurrent process could claim this port; rerun on bind failure.
    mp.spawn(worker, args=(port, arguments), nprocs=2, join=True)


if __name__ == "__main__":
    launch(sys.argv[1:])
