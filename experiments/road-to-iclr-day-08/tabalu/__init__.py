"""TabALU research prototype.

The package deliberately starts with the Phase-A numerical core.  Later modules
are added only after the arithmetic extrapolation gate is passed.
"""

from .models.executor import ExecutableProgram, ProgramNode
from .models.program_search import DifferentiableProgram

__all__ = ["DifferentiableProgram", "ExecutableProgram", "ProgramNode"]
