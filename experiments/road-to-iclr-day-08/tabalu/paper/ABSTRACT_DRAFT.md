# Abstract Draft

Neural tabular models can interpolate arithmetic relations without preserving
their computation under distribution shift. We study a modular alternative
that discovers short programs and executes protected typed primitives exactly.
On 100 matched short-chain tasks, compiled programs remain exact at 8× input
magnitude; in a causal oracle-graph ablation, learned neural primitives degrade
from 0.130 to 6.475 normalized RMSE while exact primitives remain exact.
However, the broader architecture does not survive falsification: program
recovery falls to 6.7% at depth 8, a neural residual harms 4× extrapolation, and
a season router catastrophically fails on a source-pinned real temporal
dataset. These results isolate exact execution as useful once a computation is
known, while showing that discovery and shift-safe composition—not arithmetic
implementation—remain the central unsolved problems.
