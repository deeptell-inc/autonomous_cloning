"""Closed-form autonomous cloners (star of Chen 2006, Theorems 1-2 on K_{2,m}, the diamond).

Each constructor returns a dict with keys n, a, targets, edges, J, g, t, k, M, name.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import brentq, fsolve
from .core import graph_hamiltonian, embed_dicke, site_fidelities, fan_optimal


def star_chen(M: int, J: float = 1.0) -> dict:
    """XXZ star with Dicke leaves reaching the optimal PCC bound (Chen, Cheng, Wang, Du 2006, eqs 13-14).
    The uniform field B of the original solution is a pure phase (equivalent to the Z readout correction)
    and is omitted."""
    n = M + 1; a = 0; targets = list(range(1, n)); edges = [(0, i) for i in targets]
    if M % 2 == 1:
        k, lam, t = (M - 1) // 2, np.sqrt(0.75 * (M + 1)**2 + 1), np.pi / (M + 1)
    else:
        k, lam, t = M // 2, np.sqrt(M * (M + 2)), np.pi / np.sqrt(2 * M * (M + 2))
    return dict(name=f"star M={M}", n=n, a=a, targets=targets, edges=edges, J=[J] * M, g=[lam * J] * M, t=t / J, k=k, M=M)


def k2m_odd(M: int, n_lad: int = 1, n_prime: int = 2, J: float = 1.0) -> dict:
    """Theorem 1: K_{2,m}, m = M-1 even, staggered anisotropy g_{ai} = -g_{if} = g.
    t = n pi/(2 J sqrt(k(k+1))) (n odd), g^2/J^2 = 4 (n'/n)^2 k(k+1) - k(k+1) + 1."""
    assert M % 2 == 1 and n_lad % 2 == 1
    m = M - 1; k = m // 2; n = m + 2; a = 0; mid = list(range(1, m + 1)); f = m + 1
    g2 = 4 * (n_prime / n_lad)**2 * k * (k + 1) - k * (k + 1) + 1
    assert g2 > 0, "choose n' > n sqrt((k^2+k-1)/(4k(k+1)))"
    g = J * np.sqrt(g2); t = n_lad * np.pi / (2 * J * np.sqrt(k * (k + 1)))
    edges = [(a, i) for i in mid] + [(i, f) for i in mid]
    return dict(name=f"K2m odd M={M} ({n_lad},{n_prime})", n=n, a=a, targets=mid + [f], edges=edges,
                J=[J] * (2 * m), g=[g] * m + [-g] * m, t=t, k=k, M=M)


def even_constants(k: int):
    """Theorem 2 constants (J_p = 1): B = k(k+1)m^2, A = B + M, g_p = sqrt A - sqrt B,
    g_c = ((2k-1) sqrt A - (2k+1) sqrt B)/(2k)."""
    m = 2 * k - 1; M = 2 * k; B = k * (k + 1) * m**2; A = B + M
    gp = np.sqrt(A) - np.sqrt(B); gc = ((2 * k - 1) * np.sqrt(A) - (2 * k + 1) * np.sqrt(B)) / (2 * k)
    return A, B, gp, gc


def _k2m_even_graph(M):
    m = M - 1; n = m + 2; a = 0; mid = list(range(1, m + 1)); f = m + 1
    edges = [(a, i) for i in mid] + [(i, f) for i in mid] + list(itertools.combinations(mid, 2))
    return n, a, mid, f, edges


def k2m_even_block_gaps(M, Jq):
    """Eigenvalue gaps of the k-excitation S_m-symmetric 4x4 block for K_{2,m}+ZZ chord (J_p=1)."""
    k = M // 2; m = M - 1; _, _, gp, gc = even_constants(k); gq = Jq + gc
    r = np.sqrt((k - 1) * (k + 1))
    def diag(na, nm, nf):
        za, zf = 1 - 2 * na, 1 - 2 * nf; Sm = m - 2 * nm
        return gp / 2 * za * Sm + gq / 2 * zf * Sm + gc / 2 * (Sm**2 - m) / 2
    H = np.zeros((4, 4))
    for i, lab in enumerate([(0, k, 0), (0, k - 1, 1), (1, k - 1, 0), (1, k - 2, 1)]):
        H[i, i] = diag(*lab)
    H[0, 1] = H[1, 0] = k * Jq; H[0, 2] = H[2, 0] = k; H[1, 3] = H[3, 1] = r; H[2, 3] = H[3, 2] = r * Jq
    return np.diff(np.linalg.eigvalsh(H))


def k2m_even(M: int, n1: int = 1, n2: int = 2, which: int = 0, Jrange=(-14, 14), npts=1401) -> dict:
    """Theorem 2: K_{2,m}, m = M-1 odd, with middle ZZ couplings; (J_q, t) from the PST ladder
    (E2-E1) t = n1 pi (n1 odd), (E3-E2) t = n2 pi (n2 even).  `which` selects among the roots (sorted by t)."""
    assert M % 2 == 0 and n1 % 2 == 1 and n2 % 2 == 0
    k = M // 2; _, _, gp, gc = even_constants(k); n, a, mid, f, edges = _k2m_even_graph(M); m = M - 1
    Js = np.linspace(*Jrange, npts)
    fr = lambda Jq: k2m_even_block_gaps(M, Jq)[1] / k2m_even_block_gaps(M, Jq)[0] - n2 / n1
    vals = [fr(x) for x in Js]; roots = []
    for i in range(len(Js) - 1):
        if vals[i] * vals[i + 1] < 0:
            try:
                roots.append(brentq(fr, Js[i], Js[i + 1], xtol=1e-13))
            except ValueError:
                pass
    members = sorted(((n1 * np.pi / k2m_even_block_gaps(M, Jq)[0], Jq) for Jq in roots))
    t, Jq = members[which]
    J = [1.0] * m + [Jq] * m + [0.0] * (m * (m - 1) // 2)
    g = [gp] * m + [Jq + gc] * m + [gc] * (m * (m - 1) // 2)
    return dict(name=f"K2m even M={M} ({n1},{n2})#{which}", n=n, a=a, targets=mid + [f], edges=edges, J=J, g=g, t=t, k=k, M=M, Jq=Jq)


DIAMOND_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]   # middle 0,1 (chord 0-1); input 2; far 3


def diamond(ladder=(1, 2, 2), p: float = 1.0) -> dict:
    """Uniform-anisotropy diamond K4-e (Appendix B): solve dark-state + PST ladder + revival for (q, c, lam, t).
    ladder = (n1 odd, n2 even, n_b).  Seeds come from the verified solutions."""
    seeds = {(1, 2, 2): (2.985643, 2.048572, -1.342497, 1.113388), (1, 2, 1): (2.261789, 3.401668, 0.192526, 1.279203),
             (1, 2, 3): (1.143354, 2.459968, -3.185764, 1.799182), (3, 2, 2): (-0.815199, -1.849057, 0.313848, 4.764515),
             (1, 4, 2): (1.745035, -0.998527, -1.427615, 1.880129)}
    from .blocks import diamond_blocks, dark_det, krylov_plane_gap
    n1, n2, nb = ladder
    def eqs(x):
        q, c, lam, t = x
        Ha, Hb = diamond_blocks(1.0, q, c, lam); Ea = np.linalg.eigvalsh(Ha)
        return [dark_det(Hb), (Ea[1] - Ea[0]) * t - n1 * np.pi, (Ea[2] - Ea[1]) * t - n2 * np.pi, krylov_plane_gap(Hb) * t - nb * 2 * np.pi]
    x0 = seeds.get(tuple(ladder), (2.0, 2.0, -1.0, 1.5))
    x, info, ier, msg = fsolve(eqs, x0, full_output=True, xtol=1e-13)
    assert ier == 1 and np.linalg.norm(eqs(x)) < 1e-9, f"diamond ladder {ladder} did not converge: {msg}"
    q, c, lam, t = x
    J = [c * p, p, q * p, p, q * p]
    return dict(name=f"diamond {ladder}", n=4, a=2, targets=[0, 1, 3], edges=DIAMOND_EDGES, J=J, g=[lam * x for x in J], t=t / p, k=2, M=3)


def diamond_rydberg(p: float = 1.7451728672078997, ladder=(1, 2, 2)) -> dict:
    """Diamond under the Rydberg constraint g_e = kappa J_e^2 (common kappa > 0): exact member of the
    edge-wise anisotropy family (Appendix B).  Scale symmetry fixes p; solve (q, c, kappa, t)."""
    from .blocks import diamond_blocks, dark_det, krylov_plane_gap
    n1, n2, nb = ladder
    def eqs(x):
        q, c, kappa, t = x; J = [c, p, q, p, q]
        Ha, Hb = diamond_blocks(p, q, c, None, g=[kappa * v * v for v in J]); Ea = np.linalg.eigvalsh(Ha)
        return [dark_det(Hb), (Ea[1] - Ea[0]) * t - n1 * np.pi, (Ea[2] - Ea[1]) * t - n2 * np.pi, krylov_plane_gap(Hb) * t - nb * 2 * np.pi]
    x0 = (1.3075099660487246, 2.3446585190066247, 0.15894088966389996, 4.932526279055742)
    x, info, ier, msg = fsolve(eqs, x0, full_output=True, xtol=1e-13)
    assert ier == 1 and np.linalg.norm(eqs(x)) < 1e-9, f"Rydberg diamond did not converge: {msg}"
    q, c, kappa, t = x; J = [c, p, q, p, q]
    return dict(name="diamond Rydberg", n=4, a=2, targets=[0, 1, 3], edges=DIAMOND_EDGES, J=J, g=[kappa * v * v for v in J], t=t, k=2, M=3, kappa=kappa)


def evaluate(S: dict, nphi=12) -> dict:
    """Clone fidelities of a family member S; returns dict with per-site F, min, mean, Fan."""
    H = graph_hamiltonian(S["n"], S["edges"], S["J"], S["g"])
    rest = embed_dicke(S["n"], S["a"], S["targets"], S["k"])
    fd = site_fidelities(H, S["n"], S["a"], S["targets"], rest, S["t"], nphi)
    return dict(sites=fd, min=min(fd.values()), mean=float(np.mean(list(fd.values()))), fan=fan_optimal(S["M"]))
