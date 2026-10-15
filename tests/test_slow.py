"""Slower certification checks (a few minutes): LM exactness residuals and the symbolic even-M identity."""
import numpy as np, pytest
import acloner as a
from acloner.exactness import lm_exactness_odd, lm_exactness_even
from acloner.symbolic import even_involution_identity

pytestmark = pytest.mark.slow


def test_lm_odd_k2m_from_theorem1_seed():
    S = a.k2m_odd(3)
    c, _ = lm_exactness_odd(S["n"], S["edges"], S["a"], S["targets"], S["k"], starts=1, J_init=S["J"], g_init=S["g"], t_init=S["t"])
    assert c < 1e-20


def test_lm_even_k2m_from_theorem2_seed():
    S = a.k2m_even(4)
    c, _ = lm_exactness_even(S["n"], S["edges"], S["a"], S["targets"], S["k"], starts=1, J_init=S["J"], g_init=S["g"], t_init=S["t"])
    assert c < 1e-20


def test_even_involution_identity_symbolic():
    out = even_involution_identity()
    assert max(out.values()) < 1e-25
