"""Certify exactness of an autonomous cloner by solving the revival/transfer equations with
Levenberg-Marquardt least squares (Appendix D).  residual^2 <= 1e-20 certifies an exact solution."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from .core import embed_dicke, full_state, evolve, xy_term, zz_term


def zphase(n, targets, theta):
    d = np.ones(2**n, complex)
    for s, th in zip(targets, theta):
        mask = np.array([(idx >> (n - 1 - s)) & 1 for idx in range(2**n)], bool)
        d[mask] *= np.exp(1j * th)
    return d


class EdgeBasis:
    """Precomputed XY and ZZ operators per edge so that H = sum_e J_e XY_e + g_e ZZ_e is a cheap linear combination."""
    def __init__(self, n, edges):
        self.n, self.edges = n, edges
        self.XY = [xy_term(i, j, n) for i, j in edges]; self.ZZ = [zz_term(i, j, n) for i, j in edges]

    def H(self, J, g):
        return sum(Je * X for Je, X in zip(J, self.XY)) + sum(ge * Z for ge, Z in zip(g, self.ZZ))


E0, E1 = np.array([1, 0], complex), np.array([0, 1], complex)


def lm_exactness_odd(n, edges, a, targets, k, assign="freeze0", starts=8, max_nfev=20000, seed=0, J_init=None, g_init=None, t_init=None):
    """Odd-M structure.  freeze0: |0>_a branch returns to itself, |1>_a branch -> |0>_a|D_{k+1}>;
    freeze1: the dual (|1>_a frozen, |0>_a -> |1>_a|D_{k-1}>).  Unknowns: J_e, g_e, t, 2 global phases,
    one local Z phase per target.  Returns (best residual^2, parameters dict)."""
    rng = np.random.default_rng(seed); eb = EdgeBasis(n, edges); ne = len(edges); M = len(targets)
    rk = embed_dicke(n, a, targets, k); va, vb = full_state(E0, rk, n, a), full_state(E1, rk, n, a)
    if assign == "freeze0":
        wa, wb = va, full_state(E0, embed_dicke(n, a, targets, k + 1), n, a)
    else:
        wa, wb = full_state(E1, embed_dicke(n, a, targets, k - 1), n, a), vb

    def res(x):
        H = eb.H(x[:ne], x[ne:2 * ne]); t = x[2 * ne]; z = zphase(n, targets, x[2 * ne + 3:2 * ne + 3 + M])
        ra = evolve(H, va, t) - np.exp(1j * x[2 * ne + 1]) * z * wa
        rb = evolve(H, vb, t) - np.exp(1j * x[2 * ne + 2]) * z * wb
        return np.concatenate([ra.real, ra.imag, rb.real, rb.imag])
    best = None
    for s in range(starts):
        if s == 0 and J_init is not None:
            x0 = np.concatenate([J_init, g_init, [t_init, 0, 0], np.zeros(M)])
        else:
            x0 = np.concatenate([rng.uniform(0.2, 2, ne), rng.uniform(-3, 3, ne), [rng.uniform(0.5, 8), 0, 0], rng.uniform(-np.pi, np.pi, M)])
        r = least_squares(res, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        c = float(np.sum(r.fun**2))
        if best is None or c < best[0]:
            best = (c, r.x)
    c, x = best
    return c, dict(J=x[:ne].tolist(), g=x[ne:2 * ne].tolist(), t=float(abs(x[2 * ne])))


def lm_exactness_even(n, edges, a, targets, k, starts=8, max_nfev=20000, seed=0, J_init=None, g_init=None, t_init=None, free_edges=None):
    """Even-M structure (balanced mixing, chi = pi/4, shared relative phase delta):
    U v_alpha = (e^{i p1} Z v_alpha + e^{i p2} Z |1>_a|D_{k-1}>)/sqrt2,
    U v_beta  = (e^{i(p1+d)} Z |0>_a|D_{k+1}> + e^{i(p2+d)} Z v_beta)/sqrt2.
    `free_edges`: indices of edges whose J is optimised (others keep J_init); all g are free."""
    rng = np.random.default_rng(seed); eb = EdgeBasis(n, edges); ne = len(edges); M = len(targets)
    rk = embed_dicke(n, a, targets, k); va, vb = full_state(E0, rk, n, a), full_state(E1, rk, n, a)
    ta2 = full_state(E1, embed_dicke(n, a, targets, k - 1), n, a); tb1 = full_state(E0, embed_dicke(n, a, targets, k + 1), n, a)
    s2 = 1 / np.sqrt(2); fe = list(range(ne)) if free_edges is None else list(free_edges)
    Jbase = np.array(J_init if J_init is not None else [1.0] * ne, float)

    def unpack(x):
        J = Jbase.copy(); J[fe] = x[:len(fe)]; g = x[len(fe):len(fe) + ne]; rest = x[len(fe) + ne:]
        return J, g, rest[0], rest[1], rest[2], rest[3], rest[4:4 + M]

    def res(x):
        J, g, t, p1, p2, dl, th = unpack(x); H = eb.H(J, g); z = zphase(n, targets, th)
        ra = evolve(H, va, t) - s2 * (np.exp(1j * p1) * z * va + np.exp(1j * p2) * z * ta2)
        rb = evolve(H, vb, t) - s2 * (np.exp(1j * (p1 + dl)) * z * tb1 + np.exp(1j * (p2 + dl)) * z * vb)
        return np.concatenate([ra.real, ra.imag, rb.real, rb.imag])
    best = None
    for s in range(starts):
        if s == 0 and J_init is not None and g_init is not None:
            x0 = np.concatenate([np.array(J_init)[fe], g_init, [t_init, 0, 0, 0], np.zeros(M)])
        else:
            x0 = np.concatenate([rng.uniform(0.2, 3, len(fe)), rng.uniform(-6, 6, ne), [rng.uniform(0.3, 6), 0, 0, 0], rng.uniform(-np.pi, np.pi, M)])
        r = least_squares(res, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        c = float(np.sum(r.fun**2))
        if best is None or c < best[0]:
            best = (c, r.x)
    c, x = best; J, g, t, *_ = unpack(x)
    return c, dict(J=J.tolist(), g=np.array(g).tolist(), t=float(abs(t)))
