"""Permutation-symmetric blocks of the diamond and of K_{2,m} (used by the closed-form constructions
and by the symbolic checks)."""
from __future__ import annotations
import itertools
import numpy as np
from .core import graph_hamiltonian, mm

_DIAMOND_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]


def _ket(bits, n=4):
    v = np.zeros(2**n, complex); v[int("".join(map(str, bits)), 2)] = 1; return v


_s2 = 1 / np.sqrt(2)
# 2-excitation swap(0,1)-symmetric basis: |a m_s>, |a 3>, |0 1>, |m_s 3>   (a = site 2, far = site 3)
B_ALPHA = np.array([_s2 * (_ket([1, 0, 1, 0]) + _ket([0, 1, 1, 0])), _ket([0, 0, 1, 1]), _ket([1, 1, 0, 0]),
                    _s2 * (_ket([1, 0, 0, 1]) + _ket([0, 1, 0, 1]))]).T
# 3-excitation: |a 0 1>, |a m_s 3>, |0 1 3>
B_BETA = np.array([_ket([1, 1, 1, 0]), _s2 * (_ket([1, 0, 1, 1]) + _ket([0, 1, 1, 1])), _ket([1, 1, 0, 1])]).T
V_BETA = np.array([1, np.sqrt(2), 0]) / np.sqrt(3)       # |1>_a |D_2> in B_BETA
R_MIRROR = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], float)


def diamond_blocks(p, q, c, lam, g=None):
    """(H_alpha 4x4 transfer block, H_beta 3x3 freeze block) of the diamond; g per edge overrides uniform lam."""
    J = [c, p, q, p, q]
    H = graph_hamiltonian(4, _DIAMOND_EDGES, J, [lam * x for x in J] if g is None else g)
    return mm(mm(B_ALPHA.conj().T, H), B_ALPHA).real, mm(mm(B_BETA.conj().T, H), B_BETA).real


def dark_det(Hb):
    """Dark-state condition: v_beta lies in a 2-dim Krylov space  <=>  det[v, Hv, H^2 v] = 0."""
    return np.linalg.det(np.array([V_BETA, Hb @ V_BETA, Hb @ Hb @ V_BETA]).T)


def krylov_plane_gap(Hb):
    K = np.linalg.qr(np.array([V_BETA, Hb @ V_BETA]).T)[0]
    E = np.linalg.eigvalsh(K.T @ Hb @ K); return E[1] - E[0]


def sym_basis(n, a, mid, f, nexc):
    """S_m-symmetric basis vectors labelled by occupations (n_a, n_mid, n_f) with n_a + n_mid + n_f = nexc."""
    labels, vecs = [], []; m = len(mid)
    for na in (0, 1):
        for nf in (0, 1):
            nm = nexc - na - nf
            if nm < 0 or nm > m:
                continue
            v = np.zeros(2**n, complex)
            for combo in itertools.combinations(mid, nm):
                bits = [0] * n; bits[a] = na; bits[f] = nf
                for s in combo:
                    bits[s] = 1
                v[int("".join(map(str, bits)), 2)] = 1
            vecs.append(v / np.linalg.norm(v)); labels.append((na, nm, nf))
    return labels, np.array(vecs).T
