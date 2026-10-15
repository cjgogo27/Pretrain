"""Meaningful CPU check: uninterrupted == resumed, and changed data is rejected."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from train import TinyLM

HERE = Path(__file__).resolve().parent

def execute(arguments, ok=True, expected_error="fingerprint mismatch"):
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    # Native Gloo diagnostics may still use the Windows code page. Preserve the
    # numeric/ASCII error evidence without crashing a reader thread on decoding.
    result = subprocess.run([sys.executable, *map(str, arguments)], capture_output=True, text=True,
                            encoding="utf8", errors="replace", env=environment, timeout=180)
    if ok and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if not ok:
        assert result.returncode != 0 and expected_error in result.stderr, result.stderr
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
                  "--eval-every", 6, "--eval-windows", 4, "--warmup", 3]
        execute([*common, "--out", root / "continuous"])
        execute([*common, "--out", root / "paused", "--stop-after", 6])
        execute([*common, "--out", root / "resumed", "--resume", root / "paused/latest.pt"])
        a = torch.load(root / "continuous/latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(root / "resumed/latest.pt", map_location="cpu", weights_only=False)
        for key in ["model", "optimizer", "step", "tokens", "rng_states", "data_hashes", "code_sha256"]:
            equal(a[key], b[key], key)
        expected = 12 * 2 * 16 * 2
        assert a["tokens"] == expected
        # Copy the complete executable training code, then change the actual LR.
        changed_code = root / "changed_train.py"
        original = (HERE / "train.py").read_text(encoding="utf8")
        modified = original.replace("lr = learning_rate(step + 1, args)", "lr = learning_rate(step + 1, args) * 1.01")
        assert modified != original
        changed_code.write_text(modified, encoding="utf8")
        execute([changed_code, *common[1:], "--out", root / "changed-code", "--resume", root / "paused/latest.pt"],
                ok=False, expected_error="Training code identity mismatch")
        for key, replacement, message in [("code_sha256", None, "Training code identity mismatch"),
                                           ("world_size", 2, "world_size changed"),
                                           ("torch_version", "not-the-installed-version", "PyTorch version changed")]:
            state = torch.load(root / "paused/latest.pt", weights_only=False, map_location="cpu")
            if replacement is None:
                del state[key]
            else:
                state[key] = replacement
            snapshot = root / (key + ".pt")
            torch.save(state, snapshot)
            execute([*common, "--out", root / (key + "-rejected"), "--resume", snapshot], ok=False, expected_error=message)
        execute([*common, "--lr", "0.0004", "--out", root / "plan-rejected", "--resume", root / "paused/latest.pt"],
                ok=False, expected_error="Training plan mismatch")
        with (data / "train.txt").open("a", encoding="utf8") as handle:
            handle.write("数据已改变。")
        execute([*common, "--out", root / "rejected", "--resume", root / "paused/latest.pt"], ok=False)
        new_run = execute([*common, "--steps", 2, "--warmup", 1, "--out", root / "new-stage",
                           "--init-weights", root / "paused/latest.pt"])
        startup = json.loads(new_run.stdout.splitlines()[0])
        assert startup["start_step"] == 0 and startup["start_tokens"] == 0
        new_state = torch.load(root / "new-stage/latest.pt", weights_only=False, map_location="cpu")
        assert new_state["initialization"]["kind"] == "weights_only"
        assert new_state["initialization"]["source_step"] == 6
        assert new_state["step"] == 2 and new_state["tokens"] == 2 * 2 * 16 * 2
        assert all(item["step"].item() == 2 for item in new_state["optimizer"]["state"].values())
        print(json.dumps({"status": "PASS", "checks": ["causal_future_invariance", "output_shape", "exact_cpu_model_and_optimizer_resume",
                          "rng_and_sampler_resume", "global_token_count", "changed_data_rejected",
                          "changed_lr_code_rejected", "missing_code_identity_rejected", "changed_world_size_rejected",
                          "changed_torch_version_rejected", "changed_plan_rejected", "weights_only_new_run_resets_state"],
                          "steps": 12, "tokens": expected, "torch": str(torch.__version__)}))

if __name__ == "__main__":
    main()
