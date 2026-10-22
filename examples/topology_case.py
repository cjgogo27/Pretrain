"""Offline logical 5D MoE layouts and routing counts; no GPU/collective execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path


def groups(shape):
    """Row-major explicit teaching order; not a framework's default rank order."""
    axes = list(shape)
    coordinates = list(itertools.product(*(range(shape[key]) for key in axes)))
    result = {}
    for axis in axes:
        grouped = defaultdict(list)
        for rank, coordinate in enumerate(coordinates):
            key = tuple(value for name, value in zip(axes, coordinate) if name != axis)
            grouped[key].append(rank)
        result[axis] = list(grouped.values())
        flat = [rank for group in result[axis] for rank in group]
        assert sorted(flat) == list(range(len(coordinates)))
        assert all(len(group) == shape[axis] for group in result[axis])
    return result, coordinates


def layout(world=32, tp=2, pp=2, cp=2, ep=8, etp=1):
    if min(world, tp, pp, cp, ep, etp) <= 0:
        raise ValueError("All dimensions must be positive")
    if world % (tp*pp*cp) or world % (etp*pp*ep):
        raise ValueError("World size must be divisible by both layer layout products")
    dp, edp = world//(tp*pp*cp), world//(etp*pp*ep)
    attention_shape = dict(PP=pp, DP=dp, CP=cp, TP=tp)
    expert_shape = dict(PP=pp, EDP=edp, EP=ep, ETP=etp)
    attention, ac = groups(attention_shape)
    experts, ec = groups(expert_shape)
    # Same physical pipeline stages, two layer-specific views of those devices.
    assert [x[0] for x in ac] == [x[0] for x in ec]
    assert attention["PP"] == experts["PP"]
    return dict(world_size=world, attention_shape=attention_shape, expert_shape=expert_shape,
                attention_groups=attention, expert_groups=experts,
                ranks=[dict(rank=i,attention=dict(zip(attention_shape,ac[i])),expert=dict(zip(expert_shape,ec[i]))) for i in range(world)],
                tokens_per_update=dict(micro_batch=2, accum=4, full_sequence=4096, independent_DP=dp,
                                       effective_targets=2*4*4096*dp),
                wrong_five_factor_product=dp*tp*pp*cp*ep)


def routing_case():
    tokens, experts, topk = 16, 8, 2
    balanced = [[i%8,(i+1)%8] for i in range(tokens)]
    concentrated = [[0,1] for _ in range(tokens)]
    capacity = math.ceil(1.25*tokens*topk/experts)
    reports = {}
    for name, routes in (("balanced", balanced), ("concentrated", concentrated)):
        counts = Counter(e for route in routes for e in route)
        loads = [counts[e] for e in range(experts)]
        assert sum(loads) == tokens*topk
        reports[name] = dict(routes=routes, assignments_per_expert=loads,
                             overflow_assignments=sum(max(0,x-capacity) for x in loads))
    assert reports["balanced"]["assignments_per_expert"] == [4]*8
    assert reports["concentrated"]["overflow_assignments"] == 22
    return dict(original_tokens=tokens, experts=experts, topk=topk, capacity_factor=1.25,
                capacity_per_expert=capacity, cases=reports,
                note="Assignment counts, not dropped original tokens; dropless implementations need separate memory/communication accounting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/topology")
    args = parser.parse_args()
    result = layout()
    rejected = False
    try:
        layout(ep=3)
    except ValueError:
        rejected = True
    assert rejected
    report = dict(status="PASS", layout=result, routing=routing_case(),
                  invalid_layout_rejected=True,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="CPU logical group/assignment enumeration only; no Megatron, MoE training, AllToAll or GPU speed result")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(dict(status=report["status"],world=32,DP=4,EP=8,EDP=2,
                         tokens=result["tokens_per_update"]["effective_targets"],wrong_product=result["wrong_five_factor_product"],
                         capacity=5,overflow_assignments=22)))


if __name__=="__main__": main()
