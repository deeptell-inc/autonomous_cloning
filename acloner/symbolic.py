"""Symbolic checks (sympy): freeze-block characteristic polynomial (Theorem 1) and the even-M
involution identity for symbolic k (Theorem 2 / Appendix C)."""
from __future__ import annotations
import sympy as sp


def freeze_block_charpoly():
    """Returns (factored characteristic polynomial, E_F^2) of [[0,s,s,0],[s,2g,0,r],[s,0,-2g,r],[0,r,r,0]]."""
    s, r, g, lam = sp.symbols('s r g lambda', real=True)
    H = sp.Matrix([[0, s, s, 0], [s, 2 * g, 0, r], [s, 0, -2 * g, r], [0, r, r, 0]])
    cp = sp.factor((H - lam * sp.eye(4)).det())
    return cp, 2 * s**2 + 2 * r**2 + 4 * g**2


def even_block_symbolic():
    """k-excitation S_m-symmetric block of K_{2,2k-1}+ZZ chord with symbolic k (J_p = 1)."""
    k = sp.symbols('k', positive=True); Jq, gp, gq, gc = sp.symbols('J_q g_p g_q g_c', real=True)
    m = 2 * k - 1
    def diag(na, nm, nf):
        za, zf = 1 - 2 * na, 1 - 2 * nf; Sm = m - 2 * nm
        return sp.expand(gp / 2 * za * Sm + gq / 2 * zf * Sm + gc / 2 * (Sm**2 - m) / 2)
    H = sp.zeros(4, 4)
    for i, lab in enumerate([(0, k, 0), (0, k - 1, 1), (1, k - 1, 0), (1, k - 2, 1)]):
        H[i, i] = diag(*lab)
    r = sp.sqrt(k - 1) * sp.sqrt(k + 1)
    H[0, 1] = H[1, 0] = k * Jq; H[0, 2] = H[2, 0] = k; H[1, 3] = H[3, 1] = r; H[2, 3] = H[3, 2] = r * Jq
    return k, Jq, gp, gq, gc, H


def even_involution_identity(kvalues=(2, 3, 4, 5, sp.Rational(7, 3)), digits=30):
    """Max |entry| of [H, R] after substituting the Theorem-2 closed form, for each k in kvalues
    (integers and a non-integer).  All should be 0 (to `digits` digits)."""
    k, Jq, gp, gq, gc, H = even_block_symbolic()
    c2, s2 = sp.symbols('c2 s2', real=True)
    wa = sp.sqrt(k + 1) / sp.sqrt(2 * k); wb = sp.sqrt(k - 1) / sp.sqrt(2 * k)
    v = sp.Matrix([1, 1, 0, 0]) / sp.sqrt(2); w = sp.Matrix([0, 0, wa, wb])
    u1 = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(2); u2 = sp.Matrix([0, 0, wb, -wa])
    P = sp.Matrix.hstack(v, w, u1, u2)
    Rb = sp.Matrix([[1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0], [1 / sp.sqrt(2), -1 / sp.sqrt(2), 0, 0], [0, 0, c2, s2], [0, 0, s2, -c2]])
    R = P * Rb * P.T
    m = 2 * k - 1; B = k * (k + 1) * m**2; A = B + 2 * k
    GP = sp.sqrt(A) - sp.sqrt(B); GC = ((2 * k - 1) * sp.sqrt(A) - (2 * k + 1) * sp.sqrt(B)) / (2 * k)
    subs = {gp: GP, gq: Jq + GC, gc: GC, c2: -sp.sqrt(A / (2 * B)), s2: -sp.sqrt((B - 2 * k) / (2 * B))}
    Cm = (H * R - R * H).subs(subs)
    out = {}
    for kk in kvalues:
        vals = []
        for i in range(4):
            for j in range(4):
                e = sp.expand(Cm[i, j].subs(k, kk))
                vals.append(abs(complex(sp.N(e.coeff(Jq, 1), digits)))); vals.append(abs(complex(sp.N(e.coeff(Jq, 0), digits))))
        out[str(kk)] = max(vals)
    return out
