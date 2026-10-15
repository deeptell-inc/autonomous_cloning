"""Rydberg constraint g_e = kappa J_e^2: does each family admit it?  -> results/rydberg.json"""
import json
import numpy as np
import acloner as a
out = {}
for name, S in [("star M=3", a.star_chen(3)), ("k2m_odd M=3", a.k2m_odd(3)), ("diamond", a.diamond()), ("k2m_even M=4", a.k2m_even(4)), ("diamond Rydberg", a.diamond_rydberg())]:
    J, g = np.array(S["J"]), np.array(S["g"]); nz = np.abs(J) > 1e-12
    kappa = g[nz] / J[nz]**2
    r = a.evaluate(S)
    out[name] = dict(kappa=kappa.tolist(), consistent=bool(np.ptp(kappa) < 1e-9 and np.all(kappa > 0)), F_min=r["min"], gap=r["fan"] - r["min"], J=S["J"], t=S["t"])
    print(f"{name:15s} kappa = {np.round(kappa, 6).tolist()}  single positive kappa: {out[name]['consistent']}  gap to Fan = {r['fan']-r['min']:.1e}")
json.dump(out, open("results/rydberg.json", "w"), indent=1)
