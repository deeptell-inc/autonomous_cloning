"""Robustness protocols (Appendix D): static coupling disorder, readout-time error, dephasing, Zeeman disorder."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply
from .core import graph_hamiltonian, embed_dicke, full_state, equatorial, evolve, reduced_1q, reduced_1q_dm, site_fidelities_from_states, op, SZ

PHIS = np.linspace(0, 2 * np.pi, 8, endpoint=False)


def _mean_fid(S, H, t):
    rest = embed_dicke(S["n"], S["a"], S["targets"], S["k"])
    outs = [evolve(H, full_state(equatorial(p), rest, S["n"], S["a"]), t) for p in PHIS]
    return float(np.mean([site_fidelities_from_states([reduced_1q(o, S["n"], s) for o in outs], PHIS) for s in S["targets"]]))


def unperturbed(S):
    return _mean_fid(S, graph_hamiltonian(S["n"], S["edges"], S["J"], S["g"]), S["t"])


def coupling_disorder(S, sigma, samples=100, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(samples):
        J = [x * (1 + sigma * rng.normal()) for x in S["J"]]; g = [x * (1 + sigma * rng.normal()) for x in S["g"]]
        vals.append(_mean_fid(S, graph_hamiltonian(S["n"], S["edges"], J, g), S["t"]))
    return float(np.mean(vals))


def timing_error(S, eps):
    H = graph_hamiltonian(S["n"], S["edges"], S["J"], S["g"])
    return 0.5 * (_mean_fid(S, H, S["t"] * (1 + eps)) + _mean_fid(S, H, S["t"] * (1 - eps)))


def zeeman_disorder(S, sigma_h, samples=100, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(samples):
        H = graph_hamiltonian(S["n"], S["edges"], S["J"], S["g"], h=rng.normal(0, sigma_h, S["n"]))
        vals.append(_mean_fid(S, H, S["t"]))
    return float(np.mean(vals))


def dephasing(S, gamma):
    """Lindblad d rho/dt = -i[H,rho] + gamma sum_i (Z_i rho Z_i - rho), vectorised Liouvillian."""
    n = S["n"]; d = 2**n; H = csr_matrix(graph_hamiltonian(n, S["edges"], S["J"], S["g"])); I = identity(d, format='csr')
    L = -1j * (kron(I, H) - kron(H.T, I))
    for i in range(n):
        Z = csr_matrix(op(SZ, i, n)); L = L + gamma * (kron(Z.T, Z) - kron(I, I))
    rest = embed_dicke(n, S["a"], S["targets"], S["k"]); rhos = []
    for phi in PHIS:
        v = full_state(equatorial(phi), rest, n, S["a"]); r0 = np.outer(v, v.conj()).reshape(-1, order='F')
        rhos.append(expm_multiply(L * S["t"], r0).reshape(d, d, order='F'))
    return float(np.mean([site_fidelities_from_states([reduced_1q_dm(r, n, s) for r in rhos], PHIS) for s in S["targets"]]))
