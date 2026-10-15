"""Evaluate every closed-form family for M = 3..MMAX and write results/families.json (input of Fig. 2)."""
import json, sys, time
import acloner as a
MMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = {}
for M in range(3, MMAX + 1):
    row = {"fan": a.fan_optimal(M), "single_exc_bound": a.single_excitation_bound(M)}
    fams = [("star", a.star_chen(M))]
    fams.append(("k2m_odd", a.k2m_odd(M)) if M % 2 else ("k2m_even", a.k2m_even(M)))
    if M == 3:
        fams.append(("diamond", a.diamond()))
    for name, S in fams:
        t0 = time.time(); r = a.evaluate(S)
        row[name] = dict(F_min=r["min"], F_mean=r["mean"], gap=r["fan"] - r["min"], t=S["t"], J=S["J"], g=S["g"], sec=time.time() - t0)
        print(f"M={M:2d} {name:9s} F_min={r['min']:.12f} Fan={r['fan']:.12f} gap={r['fan']-r['min']:.1e} ({time.time()-t0:.1f}s)", flush=True)
    out[M] = row
json.dump(out, open("results/families.json", "w"), indent=1)
