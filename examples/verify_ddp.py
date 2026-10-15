"""Compare fixed-sample single-process and two-rank CPU DDP gradients/one update.

An explicit legacy TCPStore avoids torchrun's implicit libuv requirement. It is a numerical
test, not a throughput benchmark or validation of CUDA/multi-machine training.
"""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from train import TinyLM, evaluate, validation_identity

def fixture():
    generator = torch.Generator().manual_seed(314159)
    stream = torch.randint(0, 257, (1200,), generator=generator)
    windows = torch.stack([stream[i:i + 11] for i in range(100)])
    return windows[:, :-1], windows[:, 1:], stream

def model():
    torch.manual_seed(42)
    return TinyLM(16, 1, 4, 10)

def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    store = dist.TCPStore("127.0.0.1", rendezvous, 2, rank == 0,
                          timeout=timedelta(seconds=40), use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        net = model()
        wrapped = DDP(net)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)
        x, y, stream = fixture()
        # Unequal valid-token counts: 10*10=100 versus 90*10=900.
        chunk = slice(0, 10) if rank == 0 else slice(10, 100)
        summed_loss = F.cross_entropy(wrapped(x[chunk]).reshape(-1, 257), y[chunk].reshape(-1), reduction="sum")
        # Standard DDP averages rank gradients, so multiply by world/global_count.
        (summed_loss * 2 / y.numel()).backward()
        gradients = {key: p.grad.detach().clone() for key, p in net.named_parameters()}
        optimizer.step()
        args = SimpleNamespace(seed=42, batch_size=4, seq_len=10, eval_windows=8)
        val = evaluate(net, stream, args, rank, 2, torch.device("cpu"))
        torch.save({"gradients": gradients, "model": net.state_dict(), "val_loss": val,
                    "eval_identity": validation_identity(stream, args)}, Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()

def compare(reference, result):
    max_error = 0.0
    for key, expected in reference.items():
        actual = result[key]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4, msg=lambda msg: key + ': ' + msg)
        max_error = max(max_error, (actual - expected).abs().max().item())
    return max_error

def main():
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("CPU DDP test requires a PyTorch build with Gloo")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    net = model()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)
    x, y, stream = fixture()
    F.cross_entropy(net(x).reshape(-1, 257), y.reshape(-1)).backward()
    gradients = {key: p.grad.detach().clone() for key, p in net.named_parameters()}
    optimizer.step()
    args = SimpleNamespace(seed=42, batch_size=4, seq_len=10, eval_windows=8)
    val = evaluate(net, stream, args, 0, 1, torch.device("cpu"))
    with tempfile.TemporaryDirectory(prefix="pretrain-ddp-") as temporary:
        root = Path(temporary)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            rendezvous = probe.getsockname()[1]
        mp.spawn(worker, args=(rendezvous, str(root)), nprocs=2, join=True)
        results = [torch.load(root / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
        grad_error = max(compare(gradients, result["gradients"]) for result in results)
        update_error = max(compare(net.state_dict(), result["model"]) for result in results)
        for result in results:
            assert abs(val - result["val_loss"]) <= 1e-6
            assert result["eval_identity"] == validation_identity(stream, args)
        print(json.dumps({"status": "PASS", "torch": str(torch.__version__), "backend": "gloo-tcp-use_libuv_false",
                          "world_size": 2, "rank_valid_tokens": [100, 900], "atol": 1e-6, "rtol": 1e-4,
                          "gradient_max_abs_error": grad_error, "update_max_abs_error": update_error,
                          "validation_max_abs_error": max(abs(val-r["val_loss"]) for r in results),
                          "eval_windows_global": 8, "eval_indices_sha256": validation_identity(stream, args)}))

if __name__ == "__main__":
    main()
