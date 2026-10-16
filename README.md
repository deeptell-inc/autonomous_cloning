# acloner — autonomous optimal phase-covariant cloners on spin networks

Reproducibility package for *Autonomous optimal phase-covariant cloning from fractional revival
on XXZ spin networks* (H. Wakaura).  Repository: https://github.com/deeptell-inc/autonomous_cloning — archived at Zenodo, DOI 10.5281/zenodo.23392981.
Everything is dense numpy on ≤ 11 qubits;
no GPU, no external data.

## Install and verify (≈ 1 min)
```bash
pip install -e ".[dev]"
make test          # 14 fast checks: Prop. 1, Theorems 1–2, Chen star, diamond (uniform + Rydberg), charpoly
make slow          # LM exactness residuals (< 1e-20) and symbolic involution identity
make quick         # M ≤ 6 families + quick robustness + Rydberg + symbolic + figures (≈ 2 min)
make all           # full data of the paper (M ≤ 8, 200 disorder samples; ≈ 10 min)
```

## Layout
| Path | Content |
|---|---|
| `acloner/core.py` | XXZ graph Hamiltonian, Dicke states, clone fidelity with one input-independent Z correction, Fan bound |
| `acloner/families.py` | `star_chen` (Chen 2006), `k2m_odd` (Theorem 1), `k2m_even` (Theorem 2), `diamond`, `diamond_rydberg` |
| `acloner/blocks.py` | permutation-symmetric blocks (diamond dark-state / PST ladder conditions) |
| `acloner/exactness.py` | Levenberg–Marquardt certification of exact revival/transfer solutions (Appendix D) |
| `acloner/symbolic.py` | sympy: freeze-block characteristic polynomial, even-M involution identity for symbolic k |
| `acloner/robustness.py` | coupling disorder, timing error, dephasing (Lindblad), Zeeman disorder |
| `scripts/` | `verify_theorems.py` → `results/families.json` (Fig. 2); `robustness.py` → `results/robustness.json` (Fig. 3); `rydberg.py`; `symbolic_proofs.py`; `make_figures.py` |
| `tests/` | `test_core.py` (fast), `test_slow.py` (`-m slow`) |

## Conventions
`H = Σ_e [J_e/2 (XX+YY) + g_e/2 ZZ]`, site 0 = most significant bit, input qubit on vertex `a`,
targets prepared in the Dicke state `|D_k⟩`.  Fidelities are averaged over 12 equatorial inputs
and corrected by a single local Z rotation per site (never per input).  All families reach the
Fan bound to ≤ 1e-15; the robustness protocol uses 8 equatorial inputs.

## Reusing a family
```python
import acloner as a
S = a.k2m_even(6)          # Theorem 2, M = 6: dict(n, a, targets, edges, J, g, t, k, M, Jq)
a.evaluate(S)["min"]       # 0.78867513… = Fan bound for M = 6
```
