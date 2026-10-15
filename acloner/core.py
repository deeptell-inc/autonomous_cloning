"""Core numerics: spin-network Hamiltonians, Dicke states, clone fidelity.

Conventions
-----------
H = sum_{(i,j) in E} [ J_ij/2 (X_i X_j + Y_i Y_j) + g_ij/2 Z_i Z_j ]
Qubit basis: |0>, |1>, with |1> = excitation.  Site ordering: site 0 is the most significant bit.
The unknown equatorial qubit (|0> + e^{i phi}|1>)/sqrt2 sits on vertex `a`; the targets carry the
Dicke state |D_k> (k excitations).  Clone fidelity is the equatorial average of the single-site
fidelity after ONE input-independent local Z rotation per site (never a per-input correction).
"""
from __future__ import annotations
import itertools
from functools import reduce
import numpy as np

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)


def op(o: np.ndarray, k: int, n: int) -> np.ndarray:
    """Single-site operator `o` on site k of n qubits (dense)."""
    return reduce(np.kron, [o if j == k else I2 for j in range(n)])


def op2(o1, i, o2, j, n):
    """Two-site operator o1_i o2_j (built by kron, no BLAS matmul)."""
    return reduce(np.kron, [o1 if k == i else o2 if k == j else I2 for k in range(n)])


def xy_term(i, j, n):
    return (op2(SX, i, SX, j, n) + op2(SY, i, SY, j, n)) / 2


def zz_term(i, j, n):
    return op2(SZ, i, SZ, j, n) / 2


def mm(A, B):
    """Complex matrix product via np.dot (numpy's `@` on Accelerate raises spurious FP warnings)."""
    return np.dot(A, B)


def graph_hamiltonian(n, edges, J, g, h=None):
    """Dense XXZ Hamiltonian; J, g are per-edge lists; h optional local fields (h_i Z_i / 2)."""
    H = np.zeros((2**n, 2**n), complex)
    for (i, j), Je, ge in zip(edges, J, g):
        H += Je * xy_term(i, j, n) + ge * zz_term(i, j, n)
    if h is not None:
        for i, hi in enumerate(h):
            H += hi / 2 * op(SZ, i, n)
    return H


def dicke(m: int, k: int) -> np.ndarray:
    """Symmetric state of m qubits with k excitations (site 0 = MSB)."""
    v = np.zeros(2**m, complex)
    for pos in itertools.combinations(range(m), k):
        v[sum(1 << (m - 1 - p) for p in pos)] = 1
    return v / np.linalg.norm(v)


def embed_dicke(n, a, targets, k):
    """State of the n-1 non-input qubits: Dicke-k on `targets`, |0> elsewhere (site order kept)."""
    m = len(targets); d = dicke(m, k)
    others = [i for i in range(n) if i != a]
    tpos = [others.index(s) for s in targets]
    v = np.zeros(2**(n - 1), complex)
    for idx in range(2**m):
        if d[idx] == 0:
            continue
        bits = [(idx >> (m - 1 - p)) & 1 for p in range(m)]
        full = 0
        for p, b in zip(tpos, bits):
            full |= b << (n - 2 - p)
        v[full] = d[idx]
    return v


def full_state(qubit, rest, n, a):
    """Tensor the single-qubit state `qubit` into position a of the (n-1)-qubit state `rest`."""
    T = rest.reshape([2] * (n - 1))
    return np.moveaxis(np.tensordot(qubit, T, axes=0), 0, a).reshape(-1)


def equatorial(phi):
    return np.array([1, np.exp(1j * phi)], complex) / np.sqrt(2)


def evolve(H, v, t):
    w, V = np.linalg.eigh(H)
    return mm(V * np.exp(-1j * w * t), mm(V.conj().T, v))


def reduced_1q(vec, n, keep):
    T = vec.reshape([2] * n)
    others = [i for i in range(n) if i != keep]
    return np.tensordot(T, T.conj(), axes=(others, others))


def reduced_1q_dm(rho, n, keep):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    R = rho.reshape([2] * n + [2] * n)
    s_in = ''.join(letters[i] for i in range(n)) + ''.join(letters[i] if i != keep else letters[n] for i in range(n))
    return np.einsum(s_in + '->' + letters[keep] + letters[n], R)


def site_fidelities_from_states(reduced_by_phi, phis):
    """reduced_by_phi: list of 1-qubit reduced states for the inputs `phis`.
    F = <rho00|a|^2 + rho11|b|^2> + 2|<a* b rho01>|  (one common Z correction)."""
    diag, coh = [], []
    for r, phi in zip(reduced_by_phi, phis):
        psi = equatorial(phi)
        diag.append(np.real(r[0, 0]) * abs(psi[0])**2 + np.real(r[1, 1]) * abs(psi[1])**2)
        coh.append(np.conj(psi[0]) * psi[1] * r[0, 1])
    return float(np.mean(diag) + 2 * abs(np.mean(coh)))


def site_fidelities(H, n, a, targets, rest, t, nphi=12):
    """Equatorial-averaged clone fidelity of every target site after free evolution for time t."""
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    w, V = np.linalg.eigh(H)
    outs = [mm(V * np.exp(-1j * w * t), mm(V.conj().T, full_state(equatorial(p), rest, n, a))) for p in phis]
    return {s: site_fidelities_from_states([reduced_1q(o, n, s) for o in outs], phis) for s in targets}


def fan_optimal(M: int) -> float:
    """Optimal 1->M phase-covariant qubit cloning fidelity (Fan et al., PRA 65, 012304)."""
    return 0.5 + np.sqrt(M * (M + 2)) / (4 * M) if M % 2 == 0 else 0.5 + (M + 1) / (4 * M)


def single_excitation_bound(q: int) -> float:
    """Symmetric 1->q bound in the single-excitation sector (Proposition 1)."""
    return 0.5 + 1 / (2 * np.sqrt(q))
