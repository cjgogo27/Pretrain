"""Meaningful CPU check: uninterrupted == resumed, and changed data is rejected."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from train import TinyLM

HERE = Path(__file__).resolve().parent

def execute(arguments, ok=True):
    result = subprocess.run([sys.executable, *map(str, arguments)], capture_output=True, text=True, encoding="utf8")
    if ok and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if not ok:
        assert result.returncode != 0 and "fingerprint mismatch" in result.stderr, result.stderr
    return result

def equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TinyLM(32, 1, 4, 16).eval()
    x = torch.randint(0, 257, (2, 16))
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 257
    with torch.no_grad():
        first, second = model(x), model(changed)
    assert first.shape == (2, 16, 257)
    assert torch.equal(first[:, :8], second[:, :8]), "Future-token leakage"
    with tempfile.TemporaryDirectory(prefix="pretrain-resume-") as temporary:
        root = Path(temporary)
        data = root / "data"
        execute([HERE / "prepare_data.py", "--out", data])
        common = [HERE / "train.py", "--data", data, "--steps", 12, "--width", 32,
                  "--layers", 1, "--seq-len", 16, "--batch-size", 2, "--accum", 2,
                  "--eval-every", 6, "--eval-batches", 2, "--warmup", 3]
        execute([*common, "--out", root / "continuous"])
        execute([*common, "--out", root / "paused", "--stop-after", 6])
        execute([*common, "--out", root / "resumed", "--resume", root / "paused/latest.pt"])
        a = torch.load(root / "continuous/latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(root / "resumed/latest.pt", map_location="cpu", weights_only=False)
        for key in ["model", "optimizer", "step", "tokens", "rng_states", "data_hashes"]:
            equal(a[key], b[key], key)
        expected = 12 * 2 * 16 * 2
        assert a["tokens"] == expected
        with (data / "train.txt").open("a", encoding="utf8") as handle:
            handle.write("数据已改变。")
        execute([*common, "--out", root / "rejected", "--resume", root / "paused/latest.pt"], ok=False)
        print(json.dumps({"status": "PASS", "checks": ["causal_future_invariance", "output_shape", "exact_cpu_model_and_optimizer_resume",
                          "rng_and_sampler_resume", "global_token_count", "changed_data_rejected"],
                          "steps": 12, "tokens": expected, "torch": str(torch.__version__)}))

if __name__ == "__main__":
    main()
