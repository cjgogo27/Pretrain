"""CPU Gloo boundary migration 2->1->2 plus an injected worker process exit.

Standalone teaching harness, not train.py strict resume, torchrun elasticity,
cluster scheduling or distributed-sharded checkpoint support. Only trusted
snapshots; fixed global batches, no dropout, full replicated FP64 AdamW state.
"""
from contextlib import nullcontext
from datetime import timedelta
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from train import TinyLM, code_sha256

TOTAL_STEPS, GLOBAL_WINDOWS, MICRO_BATCH, LENGTH = 6, 8, 2, 8


def identity():
    source=Path(__file__).read_text(encoding="utf-8-sig").replace("\r\n","\n")
    return hashlib.sha256((source+code_sha256()).encode()).hexdigest()


def corpus():
    gen=torch.Generator().manual_seed(20261022)
    return torch.randint(257,(TOTAL_STEPS*GLOBAL_WINDOWS,LENGTH+1),generator=gen)


def worker(rank,world,port,folder,target,inject_exit):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    root=Path(folder)
    try:
        store=dist.TCPStore("127.0.0.1",port,world,rank==0,timeout=timedelta(seconds=35),use_libuv=False)
        dist.init_process_group("gloo",store=store,rank=rank,world_size=world,timeout=timedelta(seconds=35))
        data=corpus()
        fingerprint=hashlib.sha256(data.numpy().tobytes()).hexdigest()
        net=TinyLM(16,1,4,LENGTH).double()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1)
        start,ledger,history=0,[],[]
        if (root/"latest.pt").exists():
            state=torch.load(root/"latest.pt",map_location="cpu",weights_only=False)
            if state["code_sha256"]!=identity() or state["data_sha256"]!=fingerprint or state["torch"]!=str(torch.__version__):
                raise ValueError("Code/data/environment mismatch")
            if state["global_windows"]!=GLOBAL_WINDOWS or state["seq_len"]!=LENGTH:
                raise ValueError("Global plan mismatch")
            net.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
            start,ledger,history=state["step"],state["ledger"],state["history"]
        assert GLOBAL_WINDOWS%(world*MICRO_BATCH)==0
        accum=GLOBAL_WINDOWS//(world*MICRO_BATCH)
        wrapped=DDP(net)
        for step in range(start,target):
            ids=list(range(step*GLOBAL_WINDOWS,(step+1)*GLOBAL_WINDOWS))
            local_ids=ids[rank::world]
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accum):
                subset=local_ids[micro*MICRO_BATCH:(micro+1)*MICRO_BATCH]
                windows=data[subset]
                with wrapped.no_sync() if micro<accum-1 else nullcontext():
                    logits=wrapped(windows[:,:-1])
                    loss=F.cross_entropy(logits.reshape(-1,257),windows[:,1:].reshape(-1))
                    (loss/accum).backward()
                if inject_exit and step==2 and micro==0:
                    # Both ranks finished and recorded the first unsynchronized
                    # microbatch. More work may start before the peer sees exit.
                    (root/f"attempt-rank-{rank}.json").write_text(json.dumps(dict(rank=rank,window_ids=subset,targets=len(subset)*LENGTH)),encoding="utf8")
                    dist.barrier()
                    if rank==1:
                        (root/"injected-exit.json").write_text(json.dumps(dict(rank=rank,exit_code=86,attempt_step=step+1)),encoding="utf8")
                        os._exit(86)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            ledger.extend(ids)
            history.append(dict(step=step+1,world_size=world,micro_batch=MICRO_BATCH,accum=accum,
                                windows=ids,committed_targets=(step+1)*GLOBAL_WINDOWS*LENGTH))
            if rank==0:
                state=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step+1,
                           committed_targets=(step+1)*GLOBAL_WINDOWS*LENGTH,ledger=ledger,history=history,
                           global_windows=GLOBAL_WINDOWS,seq_len=LENGTH,code_sha256=identity(),
                           data_sha256=fingerprint,torch=str(torch.__version__),previous_world=world)
                torch.save(state,root/"pending.pt")
                os.replace(root/"pending.pt",root/"latest.pt")
            dist.barrier()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def launch(world,root,target,inject=False):
    root.mkdir(parents=True,exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0));port=probe.getsockname()[1]
    mp.spawn(worker,args=(world,port,str(root),target,inject),nprocs=world,join=True)


def maximum_error(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max((maximum_error(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max((maximum_error(x,y) for x,y in zip(a,b)),default=0.)
    assert a==b
    return 0.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/elastic-case")
    args=parser.parse_args()
    root=Path(args.out)
    if root.exists() and any(root.iterdir()): raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    baseline,migration=root/"baseline",root/"migration"
    launch(2,baseline,6)
    launch(2,migration,2)
    before=hashlib.sha256((migration/"latest.pt").read_bytes()).hexdigest()
    failure_observed=False
    try:
        launch(2,migration,3,True)
    except mp.ProcessExitedException as error:
        # Accept only the intentional rank failure, never an arbitrary timeout.
        assert error.exit_code==86,str(error)
        failure_observed=True
    assert failure_observed and (migration/"injected-exit.json").exists()
    assert hashlib.sha256((migration/"latest.pt").read_bytes()).hexdigest()==before
    failed_state=torch.load(migration/"latest.pt",weights_only=False)
    assert failed_state["step"]==2
    launch(1,migration,4)
    launch(2,migration,6)
    a=torch.load(baseline/"latest.pt",weights_only=False)
    b=torch.load(migration/"latest.pt",weights_only=False)
    errors=dict(model=maximum_error(a["model"],b["model"]),optimizer=maximum_error(a["optimizer"],b["optimizer"]))
    assert a["ledger"]==b["ledger"]==list(range(48))
    assert a["committed_targets"]==b["committed_targets"]==384
    attempts=[json.loads((migration/f"attempt-rank-{rank}.json").read_text()) for rank in range(2)]
    observed=sum(row["targets"] for row in attempts)
    assert observed==32
    replay_ids=[i for row in attempts for i in row["window_ids"]]
    assert set(replay_ids)<=set(b["history"][2]["windows"])
    report=dict(status="PASS",environment=dict(os=platform.platform(),python=platform.python_version(),torch=str(torch.__version__),precision="FP64",threads_per_process=1,backend="Gloo; TCPStore use_libuv=False"),
                code_sha256=identity(),training_module_sha256=code_sha256(),
                migration_world_sizes=[x["world_size"] for x in b["history"]],
                accum_by_update=[x["accum"] for x in b["history"]],
                global_windows_per_update=8,seq_len=8,committed_updates=6,committed_targets=384,
                committed_ledger=b["ledger"],history=b["history"],max_abs_errors=errors,atol=1e-9,rtol=1e-8,
                injected_rank_exit=86,last_committed_step_after_failure=2,checkpoint_unchanged_after_failure=True,
                observed_discarded_targets_minimum=observed,replayed_window_ids=replay_ids,
                elapsed_seconds=time.perf_counter()-begin,
                scope="Actual local CPU process exit/relaunch and boundary DP membership migration; full replicated state, fixed global data, no dropout; not torchrun elastic admission, Kubernetes/Slurm, GPU, sharded/TP/PP/CP/EP migration, write-failure recovery or bitwise guarantee")
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report))


if __name__=="__main__": main()
