"""Controlled CPU data-wait diagnosis; timings are observations, never GPU MFU.

Inject a known 4 ms wait, profile phase ranges, remove it, rerun paired benchmarks.
No claim that this artificial delay describes the default loader or real storage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import torch
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity, record_function

from train import TinyLM, VOCAB, code_sha256


def experiment(delay, trace=None):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model = TinyLM(32, 1, 4, 32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    generator = torch.Generator().manual_seed(777)
    windows = torch.randint(0, VOCAB, (23, 4, 33), generator=generator)
    times = {key: [] for key in ("data_wait", "forward", "backward", "clip_update")}

    def step(index, measured):
        # Zeroing is intentionally assigned to clip_update for this phase budget.
        start = time.perf_counter()
        with record_function("case/data_wait"):
            if delay:
                time.sleep(delay)
            x, y = windows[index, :, :-1], windows[index, :, 1:]
        end_data = time.perf_counter()
        with record_function("case/forward"):
            loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        end_forward = time.perf_counter()
        with record_function("case/backward"):
            loss.backward()
        end_backward = time.perf_counter()
        with record_function("case/clip_update"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        end_update = time.perf_counter()
        if measured:
            for key, value in zip(times, (end_data - start, end_forward - end_data,
                                         end_backward - end_forward, end_update - end_backward)):
                times[key].append(value)

    for index in range(3):
        step(index, False)
    if trace:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
            for index in range(3, 23):
                step(index, True)
                prof.step()
        prof.export_chrome_trace(str(trace))
        events = prof.key_averages()
        phases = {event.key: event.cpu_time_total / 1000 for event in events if event.key.startswith("case/")}
        operators = [dict(name=event.key, self_cpu_ms=event.self_cpu_time_total / 1000, calls=event.count,
                          self_cpu_memory_bytes=event.self_cpu_memory_usage)
                     for event in sorted(events, key=lambda item: item.self_cpu_time_total, reverse=True)[:8]]
    else:
        for index in range(3, 23):
            step(index, True)
        phases, operators = None, None
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().numpy().tobytes())
    return dict(mean_phase_ms={key: statistics.mean(values) * 1000 for key, values in times.items()},
                median_step_ms=statistics.median([sum(values) for values in zip(*times.values())]) * 1000,
                model_sha256=digest.hexdigest(), profiled_phase_total_ms=phases, top_self_cpu_events=operators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/profile")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    order = [(repeat, delay) for repeat in range(3) for delay in (0.004, 0.0)]
    random.Random(2026).shuffle(order)
    (root / "design.json").write_text(json.dumps(dict(order=order, warmup_updates=3, measured_updates=20,
                                                     note="controlled injected delay; three paired repetitions; no GPU"), indent=2), encoding="utf8")
    benchmarks = [dict(repeat=repeat, injected_wait_seconds=delay, **experiment(delay)) for repeat, delay in order]
    traces = {name: experiment(delay, root / f"{name}-trace.json") for name, delay in (("wait", 0.004), ("removed", 0.0))}
    assert len({row["model_sha256"] for row in benchmarks} | {row["model_sha256"] for row in traces.values()}) == 1
    report = dict(environment=dict(os=platform.platform(), python=platform.python_version(), torch=str(torch.__version__), threads=1),
                  code_sha256=code_sha256(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config=dict(width=32, layers=1, heads=4, seq_len=32, batch_size=4, precision="FP32", warmup_updates=3, measured_updates=20),
                  benchmarks=benchmarks, profiled=traces, model_states_identical=True,
                  median_of_repeat_step_medians_ms={str(delay): statistics.median(row["median_step_ms"] for row in benchmarks if row["injected_wait_seconds"] == delay) for delay in (0.004, 0.0)},
                  scope="CPU controlled delay case; phase timings exclude eval/checkpoint; record_function overhead remains even outside profiler; no MFU or GPU speed claim")
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"report": str(root / "report.json"), "step_ms": report["median_of_repeat_step_medians_ms"], "model_states_identical": True}))


if __name__ == "__main__":
    main()
