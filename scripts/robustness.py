"""Robustness curves for star / K_{2,m} / diamond at M = 3 -> results/robustness.json (input of Fig. 3)."""
import json, sys
import numpy as np
import acloner as a
from acloner import robustness as R
quick = "--quick" in sys.argv
systems = {"star": a.star_chen(3), "k2m": a.k2m_odd(3), "diamond": a.diamond()}
sig = [0, 0.01, 0.02, 0.05, 0.1]; eps = [0, 0.01, 0.02, 0.05, 0.1]; gam = [0, 0.001, 0.003, 0.01, 0.03]; hz = [0, 0.05, 0.1, 0.2, 0.5]
samples = 20 if quick else 200
out = {}
for name, S in systems.items():
    out[name] = dict(F0=R.unperturbed(S), sigma=sig, eps=eps, gamma=gam, sigma_h=hz,
                     disorder=[R.coupling_disorder(S, s, samples) for s in sig],
                     timing=[R.timing_error(S, e) for e in eps],
                     dephasing=[R.dephasing(S, g) for g in gam],
                     zeeman=[R.zeeman_disorder(S, h, samples) for h in hz])
    print(name, {k: np.round(v, 4).tolist() for k, v in out[name].items() if k in ("disorder", "timing", "dephasing", "zeeman")}, flush=True)
json.dump(out, open("results/robustness.json", "w"), indent=1)
