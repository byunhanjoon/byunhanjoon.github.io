import pandas as pd

from analyze_selection_tail_risk import summarize


def test_tail_summary_is_ordered():
    result = summarize(pd.Series(range(100), dtype=float))
    assert result.loss_q90 <= result.loss_q95 <= result.loss_cvar95
    assert result.loss_std > 0
