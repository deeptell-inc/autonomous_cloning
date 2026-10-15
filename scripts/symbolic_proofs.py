"""Symbolic checks: Theorem 1 freeze-block characteristic polynomial; Theorem 2 involution identity for symbolic k."""
import sympy as sp
from acloner.symbolic import freeze_block_charpoly, even_involution_identity
cp, EF2 = freeze_block_charpoly()
print("charpoly =", cp, " ; expected -lambda^2 (lambda^2 - E_F^2), E_F^2 =", EF2)
print("[H,R] max |entry| per k:", even_involution_identity())
