"""Small causal byte-language model. FP32, offline data, DDP, and strict same-plan resume.

This is teaching code, not a production large-model training recipe.
Load only trusted checkpoints: full optimizer/RNG snapshots use Python serialization.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

VOCAB = 257
TOKENIZER = "utf8-bytes-0-255,eos-256"
CORE_KEYS = ["steps", "batch_size", "seq_len", "accum", "width", "layers", "heads", "lr",
             "warmup", "weight_decay", "clip", "seed", "device", "threads"]

class TinyLM(nn.Module):
    def __init__(self, width, layers, heads, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.token = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(seq_len, width)
        # Individually construct blocks: do not clone identical initial layer parameters.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def forward(self, ids):
        length = ids.shape[1]
        if length > self.seq_len:
            raise ValueError("Input exceeds learned position table")
        hidden = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        mask = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask, is_causal=True)
        return self.head(self.norm(hidden))

def read_data(folder, seq_len):
    streams, hashes = {}, {}
    for split in ("train", "val"):
        path = folder / f"{split}.txt"
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        docs = [doc for doc in raw.decode("utf8").strip().split("\n\n") if doc.strip()]
        tokens = []
        for doc in docs:
            tokens.extend(doc.encode("utf8"))
            tokens.append(256)
        if len(tokens) <= seq_len:
            raise ValueError(f"{split} needs more than seq_len tokens")
        streams[split] = torch.tensor(tokens, dtype=torch.long)
    hashes["tokenizer"] = TOKENIZER
    return streams, hashes

def batch(stream, size, length, generator, device):
    starts = torch.randint(len(stream) - length, (size,), generator=generator)
    windows = torch.stack([stream[int(i):int(i) + length + 1] for i in starts])
    return windows[:, :-1].to(device), windows[:, 1:].to(device)

def learning_rate(step, args):
    if args.warmup and step <= args.warmup:
        return args.lr * step / args.warmup
    progress = min(1.0, max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup)))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

@torch.no_grad()
def evaluate(model, stream, args, rank, world, device):
    model.eval()
    # Reset an independent generator each evaluation; training sampler is untouched.
    generator = torch.Generator().manual_seed(args.seed + 100000 + rank)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for _ in range(args.eval_batches):
        x, y = batch(stream, args.batch_size, args.seq_len, generator, device)
        logits = model(x)
        totals[0] += F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction="sum").double()
        totals[1] += y.numel()
    if world > 1:
        dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()

def rng_state(generator, device):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "sampler": generator.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

def restore_rng(state, generator, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)

def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"

def save_checkpoint(model, optimizer, generator, step, tokens, hashes, args, rank, world, device):
    local = rng_state(generator, device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        payload = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step, "tokens": tokens, "data_hashes": hashes, "world_size": world,
                   "config": vars(args), "rng_states": states, "torch_version": str(torch.__version__),
                   "code_revision": git_revision(),
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        target = Path(args.out) / "latest.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
    if world > 1:
        dist.barrier()

@torch.no_grad()
def sample(model, device, seed):
    model.eval()
    ids = torch.tensor([list("预训练".encode("utf8"))], device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed + 200000)
    for _ in range(80):
        logits = model(ids[:, -model.seq_len:])[:, -1, :]
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
        ids = torch.cat([ids, next_id], dim=1)
    return bytes(token for token in ids[0].tolist() if token < 256).decode("utf8", errors="replace")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="runs/data")
    parser.add_argument("--out", default="runs/tiny")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    for name, default in [("steps", 100), ("batch-size", 4), ("seq-len", 64), ("accum", 1),
                          ("width", 128), ("layers", 2), ("heads", 4), ("warmup", 10),
                          ("eval-every", 20), ("eval-batches", 4), ("seed", 42), ("threads", 1)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in [("lr", 3e-4), ("weight-decay", 0.1), ("clip", 1.0)]:
        parser.add_argument(f"--{name}", type=float, default=default)
    args = parser.parse_args()
    for key in ["steps", "batch_size", "seq_len", "accum", "width", "layers", "heads",
                "eval_every", "eval_batches", "threads"]:
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.width % args.heads or args.warmup < 0 or args.lr <= 0 or args.clip <= 0 or args.weight_decay < 0:
        parser.error("Invalid model/optimizer configuration")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be in 1..steps")
    return args

def run(args):
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.set_num_threads(args.threads)
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use CPU or install a matching CUDA build")
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    streams, hashes = read_data(Path(args.data), args.seq_len)
    raw_model = TinyLM(args.width, args.layers, args.heads, args.seq_len).to(device)
    model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else raw_model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 1000 + rank)
    step, tokens = 0, 0
    if args.resume:
        # Trusted local full snapshot only; contains optimizer and Python RNG objects.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["data_hashes"] != hashes:
            raise ValueError("Data/tokenizer fingerprint mismatch; resume refused")
        if state["world_size"] != world:
            raise ValueError("world_size changed; strict resume refused")
        differences = [key for key in CORE_KEYS if state["config"][key] != getattr(args, key)]
        if differences:
            raise ValueError(f"Training plan mismatch; resume refused: {differences}")
        if state["torch_version"] != str(torch.__version__):
            raise ValueError("PyTorch version changed; strict resume refused")
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, tokens = state["step"], state["tokens"]
        restore_rng(state["rng_states"][rank], generator, device)
    target_step = args.stop_after or args.steps
    if target_step <= step:
        raise ValueError("Target step must exceed the restored step")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_tokens, started = tokens, time.perf_counter()
    if rank == 0:
        print(json.dumps({"parameters": sum(p.numel() for p in raw_model.parameters()),
                          "world_size": world, "device": str(device), "torch": str(torch.__version__),
                          "tokens_per_update": args.batch_size * args.seq_len * args.accum * world,
                          "start_step": step}), flush=True)
    while step < target_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step + 1, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        average_loss = torch.zeros((), device=device)
        for micro in range(args.accum):
            context = model.no_sync() if world > 1 and micro < args.accum - 1 else nullcontext()
            with context:
                x, y = batch(streams["train"], args.batch_size, args.seq_len, generator, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step + 1}, rank {rank}")
                (loss / args.accum).backward()
                average_loss += loss.detach() / args.accum
        norm = nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        tokens += args.batch_size * args.seq_len * args.accum * world
        if step % args.eval_every == 0 or step == target_step:
            if world > 1:
                dist.all_reduce(average_loss)
                average_loss /= world
            val_loss = evaluate(raw_model, streams["val"], args, rank, world, device)
            save_checkpoint(raw_model, optimizer, generator, step, tokens, hashes, args, rank, world, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            record = {"step": step, "tokens": tokens, "train_loss": average_loss.item(),
                      "val_loss": val_loss, "ppl": math.exp(min(val_loss, 80)), "lr": lr,
                      "grad_norm": norm.item(), "elapsed_seconds": elapsed,
                      "tokens_per_second": (tokens - start_tokens) / elapsed,
                      "eval_batches_per_rank": args.eval_batches}
            if rank == 0:
                with (out / "metrics.jsonl").open("a", encoding="utf8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
    if rank == 0:
        (out / "sample.txt").write_text(sample(raw_model, device, args.seed), encoding="utf8")
    if world > 1:
        dist.barrier()

if __name__ == "__main__":
    try:
        run(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
