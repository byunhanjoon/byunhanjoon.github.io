"""Phase-A TabALU model facade.

Operand inference, routing, typed heterogeneous operators, and the residual are
intentionally absent until the exact-execution gate is established.
"""

from .program_search import DifferentiableProgram


class TabALU(DifferentiableProgram):
    pass
