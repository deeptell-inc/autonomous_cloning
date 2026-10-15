"""Fast checks (< 1 min): Prop. 1, Theorem 1, Theorem 2, Chen star, diamond, symbolic charpoly."""
import numpy as np, pytest, sympy as sp
import acloner as a
from acloner.core import graph_hamiltonian, embed_dicke, site_fidelities, single_excitation_bound
from acloner.symbolic import freeze_block_charpoly

RTOL = 1e-9


def test_prop1_path_p3_single_excitation():
    # P3 with input at the centre, k=0: F_s = 1/2 + |beta|/2 with |beta| = 1/sqrt2 at t = pi/(2 sqrt2) (J=1)
    n, a_, targets = 3, 1, [0, 2]
    H = graph_hamiltonian(n, [(0, 1), (1, 2)], [1, 1], [0, 0])
    t = np.pi / (2 * np.sqrt(2))
    f = site_fidelities(H, n, a_, targets, embed_dicke(n, a_, targets, 0), t)
    for s in targets:
        assert f[s] == pytest.approx(single_excitation_bound(2), rel=RTOL)


@pytest.mark.parametrize("M", [3, 5])
def test_chen_star_odd(M):
    r = a.evaluate(a.star_chen(M)); assert r["min"] == pytest.approx(r["fan"], rel=RTOL)


@pytest.mark.parametrize("M", [4, 6])
def test_chen_star_even(M):
    r = a.evaluate(a.star_chen(M)); assert r["min"] == pytest.approx(r["fan"], rel=RTOL)


@pytest.mark.parametrize("M,n,np_", [(3, 1, 2), (5, 1, 2), (3, 3, 4)])
def test_theorem1_k2m_odd(M, n, np_):
    r = a.evaluate(a.k2m_odd(M, n, np_)); assert r["min"] == pytest.approx(r["fan"], rel=RTOL)


@pytest.mark.parametrize("M", [4, 6])
def test_theorem2_k2m_even(M):
    S = a.k2m_even(M); r = a.evaluate(S); assert r["min"] == pytest.approx(r["fan"], rel=RTOL)
    k = M // 2; A, B, gp, gc = a.even_constants(k)
    assert gp * (gp - gc) == pytest.approx(1.0, rel=1e-12)          # identity g_p (g_p - g_c) = 1


def test_diamond_exact():
    S = a.diamond(); r = a.evaluate(S)
    assert r["min"] == pytest.approx(r["fan"], rel=RTOL)
    assert S["J"][2] == pytest.approx(2.985643, abs=1e-6) and S["t"] == pytest.approx(1.113388, abs=1e-6)


def test_diamond_rydberg_exact():
    S = a.diamond_rydberg(); r = a.evaluate(S); J, g = np.array(S["J"]), np.array(S["g"])
    assert r["min"] == pytest.approx(r["fan"], rel=RTOL)
    assert S["kappa"] > 0 and np.allclose(g, S["kappa"] * J**2)


def test_freeze_block_charpoly():
    cp, EF2 = freeze_block_charpoly(); lam = sp.Symbol('lambda', real=True)
    assert sp.simplify(cp - lam**2 * (lam**2 - EF2)) == 0   # spectrum {0, 0, +E_F, -E_F}


def test_fan_values():
    assert a.fan_optimal(3) == pytest.approx(5 / 6) and a.fan_optimal(2) == pytest.approx(0.5 + np.sqrt(2) / 4)
