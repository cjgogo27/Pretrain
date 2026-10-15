"""CPU multi-step DDP: fixed-sample equivalence, then real-loop exact resume.

No AMP, sharding, elasticity or GPU performance claims. Two local Gloo ranks.
"""
from contextlib import nullcontext
from datetime import timedelta
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from train import TinyLM, code_sha256
from verify_ddp import compare
from verify_resume import equal, execute

HERE = Path(__file__).resolve().parent


def fixed_steps(rank=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    net = TinyLM(16, 1, 4, 8)
    wrapped = DDP(net) if rank is not None else net
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)
    generator = torch.Generator().manual_seed(987)
    records = []
    for step in range(5):
        windows = torch.randint(0, 257, (8, 9), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        if rank is None:
            F.cross_entropy(wrapped(windows[:, :-1]).reshape(-1, 257), windows[:, 1:].reshape(-1)).backward()
        else:
            # Every update has the identical 8 windows. DDP rank means, A=2.
            local = windows[rank::2]
            for micro in range(2):
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    part = local[2 * micro:2 * micro + 2]
                    loss = F.cross_entropy(wrapped(part[:, :-1]).reshape(-1, 257), part[:, 1:].reshape(-1))
                    (loss / 2).backward()
        gradients = {key: p.grad.clone() for key, p in net.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2, error_if_nonfinite=True)
        assert norm > 0.2, "Fixture must exercise active clipping"
        optimizer.step()
        records.append(dict(step=step + 1, gradients=gradients, norm=norm.clone(),
                            model={key: value.clone() for key, value in net.state_dict().items()}))
    return records


def fixed_worker(rank, port, output):
    try:
        store = dist.TCPStore("127.0.0.1", port, 2, rank == 0,
                              timeout=timedelta(seconds=90), use_libuv=False)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                                timeout=timedelta(seconds=90))
        torch.save(fixed_steps(rank), Path(output) / f"fixed-{rank}.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/ddp-training")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = fixed_steps()
    with tempfile.TemporaryDirectory(prefix="pretrain-ddp-training-") as temporary:
        root = Path(temporary)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        mp.spawn(fixed_worker, args=(port, str(root)), nprocs=2, join=True)
        errors = dict(gradient=0.0, parameter=0.0, preclip_norm=0.0)
        for rank in range(2):
            actual = torch.load(root / f"fixed-{rank}.pt", weights_only=True)
            for expected, result in zip(reference, actual, strict=True):
                errors["gradient"] = max(errors["gradient"], compare(expected["gradients"], result["gradients"]))
                errors["parameter"] = max(errors["parameter"], compare(expected["model"], result["model"]))
                torch.testing.assert_close(expected["norm"], result["norm"], atol=1e-6, rtol=1e-4)
                errors["preclip_norm"] = max(errors["preclip_norm"], abs(float(expected["norm"] - result["norm"])))

        data = root / "data"
        execute([HERE / "prepare_data.py", "--out", data, "--pools"])
        common = [HERE / "launch_cpu.py", "--data", data, "--steps", 8, "--width", 32,
                  "--layers", 1, "--seq-len", 16, "--batch-size", 2, "--accum", 2,
                  "--eval-every", 4, "--eval-windows", 8, "--warmup", 2, "--clip", 0.2,
                  "--domain-weight", 0.5]
        for name, extra in [("continuous", []), ("paused", ["--stop-after", 4]),
                            ("resumed", ["--resume", root / "paused/latest.pt"])]:
            result = execute([*common, "--out", root / name, *extra])
            (out / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf8")
        a = torch.load(root / "continuous/latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(root / "resumed/latest.pt", map_location="cpu", weights_only=False)
        keys = ["model", "optimizer", "step", "tokens", "rng_states", "data_hashes", "code_sha256", "domain_tokens", "initialization"]
        for key in keys:
            equal(a[key], b[key], key)
        assert a["world_size"] == 2 and a["tokens"] == 8 * 2 * 16 * 2 * 2 == 1024
        assert len(a["rng_states"]) == 2
        assert not torch.equal(a["rng_states"][0]["sampler"], a["rng_states"][1]["sampler"])
        last = {}
        for name in ("continuous", "resumed"):
            last[name] = json.loads((root / name / "metrics.jsonl").read_text().splitlines()[-1])
        for key in ("step", "tokens", "train_loss", "val_loss", "lr", "grad_norm", "eval_indices_sha256", "domain_tokens", "general_tokens"):
            assert last["continuous"][key] == last["resumed"][key], key
        assert (root / "continuous/sample.txt").read_bytes() == (root / "resumed/sample.txt").read_bytes()
        # A changed mixture is a changed training plan; a changed pool is new data.
        execute([*common, "--domain-weight", 0.2, "--out", root / "bad-plan", "--resume", root / "paused/latest.pt"],
                ok=False, expected_error="Training plan mismatch")
        with (data / "train-domain.txt").open("a", encoding="utf8") as handle:
            handle.write("池身份改变。")
        execute([*common, "--out", root / "bad-data", "--resume", root / "paused/latest.pt"], ok=False)
        report = dict(status="PASS", environment=dict(os=platform.platform(), python=platform.python_version(), torch=str(torch.__version__)),
                      code_sha256=code_sha256(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      backend="Gloo TCPStore(use_libuv=False)", fixed_sample=dict(updates=5, world_size=2, accum=2,
                      batch_per_rank=2, seq_len=8, tokens_per_update=64, clip=0.2, clipping_active_all_steps=True,
                      comparison="same windows in single global batch versus two ranks / two micros", max_abs_errors=errors, atol=1e-6, rtol=1e-4),
                      real_loop=dict(updates=8, interrupted_after=4, tokens=1024, world_size=2, accum=2,
                      batch_per_rank=2, seq_len=16, width=32, layers=1, clip=0.2, domain_weight=0.5,
                      exact_equal_fields=keys, both_rank_rng_restored=True, final_metrics_equal=True,
                      generated_sample_equal=True, changed_mixture_rejected=True, changed_pool_rejected=True,
                      final_metrics=last["continuous"]), scope="CPU FP32 local only; no CUDA/NCCL/multi-node/sharded/elastic validation")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
