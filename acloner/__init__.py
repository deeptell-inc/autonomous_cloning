"""acloner: closed-form autonomous optimal phase-covariant cloners on spin networks."""
from .core import fan_optimal, single_excitation_bound, graph_hamiltonian, site_fidelities, embed_dicke
from .families import star_chen, k2m_odd, k2m_even, diamond, diamond_rydberg, evaluate, even_constants

__all__ = ["fan_optimal", "single_excitation_bound", "graph_hamiltonian", "site_fidelities", "embed_dicke",
           "star_chen", "k2m_odd", "k2m_even", "diamond", "diamond_rydberg", "evaluate", "even_constants"]
__version__ = "0.1.0"
