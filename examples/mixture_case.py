"""Three paired seeds, two pool probabilities, fixed model/token budget/evaluation.

No external datasets. This measures template-distribution NLL, not expertise.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from types import SimpleNamespace

import torch
from train import TinyLM, code_sha256, encode_documents, evaluate, validation_identity, validation_indices

HERE = Path(__file__).resolve().parent


def execute(arguments, output):
    result = subprocess.run([sys.executable, *map(str, arguments)], capture_output=True, text=True,
                            encoding="utf8", timeout=180)
    output.write_text(result.stdout + result.stderr, encoding="utf8")
    if result.returncode:
        raise RuntimeError(f"Command failed; see {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/mixture")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    design = dict(seeds=[42, 43, 44], probabilities=[0.2, 0.5], steps=100, batch_size=4, seq_len=64,
                  accum=1, width=128, layers=2, heads=4, lr=0.0003, warmup=10, eval_windows=64,
                  total_tokens=25600, validation_seed=2026, precision="FP32", threads=1, training_eval_windows=16)
    order = [(seed, probability) for seed in design["seeds"] for probability in design["probabilities"]]
    random.Random(2026).shuffle(order)
    # Persist the design before collecting results. Three seeds are illustrative,
    # not a powered test; validation windows are not independent replications.
    (root / "design.json").write_text(json.dumps(dict(design=design, run_order=order), indent=2), encoding="utf8")
    data = root / "data"
    execute([HERE / "prepare_data.py", "--out", data, "--pools"], root / "prepare.log")
    torch.set_num_threads(1)
    validation_args = SimpleNamespace(seed=2026, batch_size=4, seq_len=64, eval_windows=64)
    streams = {pool: encode_documents((data / f"val-{pool}.txt").read_bytes()) for pool in ("domain", "general")}
    validation = {pool: dict(file_sha256=hashlib.sha256((data / f"val-{pool}.txt").read_bytes()).hexdigest(),
                            indices_sha256=validation_identity(stream, validation_args),
                            indices=validation_indices(stream, validation_args), valid_tokens=4096)
                  for pool, stream in streams.items()}
    runs = []
    for seed, probability in order:
        out = root / f"seed-{seed}-p-{probability}"
        execute([HERE / "train.py", "--data", data, "--out", out, "--seed", seed,
                 "--domain-weight", probability], root / f"seed-{seed}-p-{probability}.log")
        state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        assert state["tokens"] == design["total_tokens"] and state["code_sha256"] == code_sha256()
        model = TinyLM(128, 2, 4, 64)
        model.load_state_dict(state["model"])
        losses = {pool: evaluate(model, stream, validation_args, 0, 1, torch.device("cpu")) for pool, stream in streams.items()}
        assert state["tokens"] - state["domain_tokens"] >= 0
        runs.append(dict(seed=seed, requested_domain_probability=probability, tokens=state["tokens"],
                         domain_tokens=state["domain_tokens"], general_tokens=state["tokens"] - state["domain_tokens"],
                         actual_domain_fraction=state["domain_tokens"] / state["tokens"],
                         domain_loss=losses["domain"], general_loss=losses["general"],
                         code_sha256=state["code_sha256"], data_hashes=state["data_hashes"],
                         metrics=[json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]))
    runs.sort(key=lambda row: (row["seed"], row["requested_domain_probability"]))
    paired = []
    for seed in design["seeds"]:
        a, b = [row for row in runs if row["seed"] == seed]
        paired.append(dict(seed=seed, domain_loss_delta_b_minus_a=b["domain_loss"] - a["domain_loss"],
                           general_loss_delta_b_minus_a=b["general_loss"] - a["general_loss"]))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf8"))
    report = dict(design=design, run_order=order, environment=dict(os=platform.platform(), python=platform.python_version(), torch=str(torch.__version__)),
                  code_sha256=code_sha256(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest=manifest, validation=validation, runs=runs, paired=paired,
                  mean_paired_delta={pool: statistics.mean(row[f"{pool}_loss_delta_b_minus_a"] for row in paired) for pool in streams},
                  scope="three seeds on shared-template data; not domain capability or a universal optimal mixture")
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"report": str(root / "report.json"), "paired": paired, "mean_paired_delta": report["mean_paired_delta"]}))


if __name__ == "__main__":
    main()
