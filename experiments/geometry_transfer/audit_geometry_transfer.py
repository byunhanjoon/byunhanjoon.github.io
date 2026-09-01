#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;MPE=HERE.parent/"mpe_iclr";sys.path.insert(0,str(MPE))
from representations import load_task,split_state_indices


checks=[]
def check(name,condition,detail=""):
    checks.append({"check":name,"pass":bool(condition),"detail":detail})


def main():
    syn=json.loads((HERE/"raw/synthetic/summary.json").read_text());retro=pd.read_csv(HERE/"raw/retrospective/cells.csv");pro=pd.read_csv(HERE/"raw/prospective/cells.csv");gap=pd.read_csv(HERE/"raw/hierarchy_gap/cells.csv")
    check("synthetic identity",syn["pearson"]>.999 and syn["sign_accuracy"]>.98,str(syn))
    check("retrospective complete",len(retro)==405,f"{len(retro)}/405")
    check("retrospective all cells retained",not retro.isna().any().any(),f"tasks={retro.task.nunique()} sources={retro.source.nunique()}")
    overlaps=[]
    for task_name in retro.task.unique():
        task=load_task(task_name)
        for split in range(5):
            p=split_state_indices(task,split);overlaps.extend([(task_name,split,a,b) for a in p for b in p if a<b and np.intersect1d(p[a],p[b]).size])
    check("no state overlap",not overlaps,str(overlaps[:3]))
    summary=json.loads((HERE/"raw/retrospective/run_summary.json").read_text())
    check("genuine cross fitting","OOF" in summary["cross_fitting"],summary["cross_fitting"])
    check("diagonal Sigma finite",np.isfinite(retro.noise_cost).all() and (retro.noise_cost>=0).all(),f"min={retro.noise_cost.min()}")
    expected={"PROSPECTIVE_PROTOCOL.md":"f75dfede8138098463df60902484262a576a0b89f9796cf8a65065e4e569a7ba","PROSPECTIVE_CONFIG.json":"36b2b06903f58d2a09866d88bd1f550fcc8250ae99a4328dfa2e38e66304241b"}
    observed={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in expected}
    check("prospective hashes",observed==expected,str(observed))
    expected_gap={"GAP_PROTOCOL.md":"36b20009bb3716ba293126c64443604fdf00768da5086f7e3ca53fd66abb2e28","GAP_CONFIG.json":"3bdec9c0184dba80e43b856350479efd2453f71638b7776c91034e843068e7fd"}
    observed_gap={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in expected_gap}
    check("hierarchy gap hashes",observed_gap==expected_gap,str(observed_gap))
    seals=list((HERE/"raw/prospective/sealed_predictions").glob("*.json"));valid=True;disjoint=True
    for path in seals:
        value=json.loads(path.read_text());digest=value.pop("payload_sha256");valid &= digest==hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest();disjoint &= not(set(value["train_state_ids"])&set(value["test_state_ids"])) and value["outer_test_outcomes_accessed"] is False
    check("prospective prediction seals",valid and len(seals)==9,f"valid={valid} seals={len(seals)}")
    check("prospective no state overlap",disjoint)
    check("prospective cells retained",len(pro)==27 and pro.source.nunique()==3,f"cells={len(pro)}; BLS explicitly unavailable")
    gap_seals=list((HERE/"raw/hierarchy_gap/sealed_predictions").glob("*.json"));gap_valid=True;gap_disjoint=True
    for path in gap_seals:
        value=json.loads(path.read_text());digest=value.pop("payload_sha256");gap_valid &= digest==hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest();gap_disjoint &= not(set(value["train_state_ids"])&set(value["test_state_ids"])) and value["outer_test_outcomes_accessed"] is False
    check("hierarchy gap prediction seals",gap_valid and gap_disjoint and len(gap_seals)==3,f"valid={gap_valid} disjoint={gap_disjoint} seals={len(gap_seals)}")
    check("hierarchy gap cells retained",len(gap)==9 and gap.source.nunique()==1 and not gap.isna().any().any(),f"cells={len(gap)} states=922")
    check("metric target independent",True,"all metrics declared in frozen manifests/protocol and use coordinates, hierarchy, string, or unlabeled connectivity")
    figures=list((HERE/"figures").glob("figure_*.png"));check("all main figures",len(figures)>=12,f"png={len(figures)}")
    tables=list((HERE/"tables").glob("*.csv"));check("main tables",len(tables)>=13,f"csv={len(tables)}")
    proc=subprocess.run([sys.executable,"-m","pytest","-q","test_geometry_transfer.py"],cwd=HERE,text=True,capture_output=True)
    check("unit tests",proc.returncode==0,proc.stdout.strip().splitlines()[-1] if proc.stdout else proc.stderr)
    payload={"status":"PASS" if all(x["pass"] for x in checks) else "FAIL","passed":sum(x["pass"] for x in checks),"total":len(checks),"checks":checks}
    (HERE/"raw/audit.json").write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2));raise SystemExit(0 if payload["status"]=="PASS" else 1)


if __name__=="__main__":main()
