"""Figures from results/*.json (Arial, vector PDF)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42})
fam = json.load(open("results/families.json")); Ms = sorted(int(m) for m in fam)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(Ms, [fam[str(m)]["fan"] for m in Ms], "k-", label="Fan bound")
ax.plot(Ms, [fam[str(m)]["single_exc_bound"] for m in Ms], "k:", label="single-excitation bound")
for key, mk, lab in [("star", "o", "star"), ("k2m_odd", "s", r"$K_{2,m}$ odd"), ("k2m_even", "^", r"$K_{2,m}$ even"), ("diamond", "D", "diamond")]:
    xs = [m for m in Ms if key in fam[str(m)]]
    ax.plot(xs, [fam[str(m)][key]["F_min"] for m in xs], mk, mfc="none", label=lab)
ax.set_xlabel("M"); ax.set_ylabel(r"$F_{\min}$"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("figures/fig2_families.pdf")
try:
    rob = json.load(open("results/robustness.json"))
except FileNotFoundError:
    raise SystemExit("run scripts/robustness.py first")
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.2))
for ax, (xk, yk, xl) in zip(axs, [("sigma", "disorder", r"$\sigma_J$"), ("eps", "timing", r"$\epsilon_t$"), ("gamma", "dephasing", r"$\gamma t$"), ("sigma_h", "zeeman", r"$\sigma_h$")]):
    for name, d in rob.items():
        ax.plot(d[xk], d[yk], "o-", ms=3, label=name)
    ax.set_xlabel(xl); ax.set_ylabel("mean F")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig("figures/fig3_robustness.pdf")
print("wrote figures/fig2_families.pdf figures/fig3_robustness.pdf")
