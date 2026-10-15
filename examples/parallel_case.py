"""CPU algebra check of TP FFN splits; GPipe schedule arithmetic, no real TP/PP."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/parallel")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(6, 4, dtype=torch.float64)
    output = F.gelu(x @ w1) @ w2
    ((output - target) ** 2).mean().backward()
    split_x = x.detach().clone().requires_grad_()
    columns = [part.detach().clone().requires_grad_() for part in w1.chunk(2, dim=1)]
    rows = [part.detach().clone().requires_grad_() for part in w2.chunk(2, dim=0)]
    partial = [F.gelu(split_x @ a) @ b for a, b in zip(columns, rows)]
    split_output = sum(partial)  # Emulates forward SUM all-reduce, in one process.
    ((split_output - target) ** 2).mean().backward()
    comparisons = dict(output=(output, split_output), input_gradient=(x.grad, split_x.grad),
                       w1_gradient=(w1.grad, torch.cat([part.grad for part in columns], dim=1)),
                       w2_gradient=(w2.grad, torch.cat([part.grad for part in rows], dim=0)))
    errors = {}
    for name, (expected, actual) in comparisons.items():
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        errors[name] = (actual - expected).abs().max().item()
    # GPipe fill-drain: P=2, m=4, equal one-slot forward/backward, zero communication.
    schedule = [["F1", "F2", "F3", "F4", "-", "-", "B4", "B3", "B2", "B1"],
                ["-", "F1", "F2", "F3", "F4", "B4", "B3", "B2", "B1", "-"]]
    for micro in range(1, 5):
        assert schedule[0].index(f"F{micro}") < schedule[1].index(f"F{micro}")
        assert schedule[1].index(f"F{micro}") < schedule[1].index(f"B{micro}") < schedule[0].index(f"B{micro}")
    assert all(sum(job != "-" for job in stage) == 8 for stage in schedule)
    report = dict(status="PASS", torch=str(torch.__version__), scope="single-process FP64 algebra and ideal schedule; no distributed TP/PP or measured communication",
                  shapes=dict(x=[6, 4], w1=[4, 8], w2=[8, 4], per_shard_w1=[4, 4], per_shard_w2=[4, 4]),
                  max_abs_errors=errors, tolerance=dict(atol=1e-12, rtol=1e-12),
                  communication_example=dict(B=2, S=1024, d=4096, bytes_per_element=2, tp=2,
                                             payload_bytes=2 * 1024 * 4096 * 2,
                                             ring_send_bytes_per_rank_per_collective=2 * 1024 * 4096 * 2,
                                             ffn_forward_plus_backward_send_bytes_per_rank=2 * 2 * 1024 * 4096 * 2),
                  pipeline=dict(stages=2, microbatches=4, schedule=schedule, slots=10,
                                idle_device_slots=4, total_device_slots=20, bubble_fraction=0.2,
                                assumptions="GPipe fill-drain; equal forward/backward time; balanced stages; communication zero; optimizer excluded"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
