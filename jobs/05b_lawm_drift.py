"""Which LaWM parts did LaWAM policy pretraining change? Relative L2 drift of lawam_pretrain's lam.* vs lawam_lam.

(05_lawm_sanity.py reported 0.0 for every part: it compared against state_dict() references that load_state_dict had
already overwritten in place. This compares two independent CPU state dicts.)
"""
import json
import os

import torch

G, L = os.environ["G05_ROOT"], os.environ["LAWAM_ROOT"]
REL = f"{L}/latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
PRE = f"{L}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt"

rel = {k.replace("lam.", "", 1): v for k, v in torch.load(REL, map_location="cpu", weights_only=False)["state_dict"].items()}
pre_all = torch.load(PRE, map_location="cpu", weights_only=False)
for k in ("state_dict", "model", "module"):
    if isinstance(pre_all, dict) and k in pre_all and isinstance(pre_all[k], dict):
        pre_all = pre_all[k]
P = "policy_backend.lam."
pre = {k[len(P):]: v for k, v in pre_all.items() if k.startswith(P)}

parts = sorted({k.split(".")[0] for k in pre})
out = {"n_release": len(rel), "n_pretrain": len(pre), "only_in_release": sorted(set(rel) - set(pre))[:10],
       "only_in_pretrain": sorted(set(pre) - set(rel))[:10], "parts": {}}
for part in parts:
    num = den = 0.0
    n = changed = 0
    for k, v in pre.items():
        if k.split(".")[0] != part or k not in rel or not v.dtype.is_floating_point:
            continue
        d = (v.float() - rel[k].float()).pow(2).sum().item()
        num += d
        den += rel[k].float().pow(2).sum().item()
        n += 1
        changed += int(d > 0)
    out["parts"][part] = {"tensors": n, "tensors_changed": changed, "relative_l2_drift": (num / den) ** 0.5 if den else None}
print(json.dumps(out, indent=2))
json.dump(out, open(f"{G}/results/lawm_sanity/param_drift.json", "w"), indent=2)
