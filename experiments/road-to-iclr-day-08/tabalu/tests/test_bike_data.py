from __future__ import annotations

import pandas as pd

from tabalu.models.bike_typed import bike_typed_design


def test_bike_typed_design_is_finite_and_stable() -> None:
    frame = pd.DataFrame(
        {
            "dteday": pd.to_datetime(["2011-01-01", "2012-07-01"]),
            "temp": [0.2, 0.8],
            "atemp": [0.25, 0.75],
            "hum": [0.8, 0.4],
            "windspeed": [0.0, 0.3],
            "hr": [0, 23],
            "weekday": [6, 0],
            "mnth": [1, 7],
            "workingday": [0, 1],
            "holiday": [0, 0],
            "season": [1, 3],
            "weathersit": [1, 2],
        }
    )
    design, names = bike_typed_design(frame)
    assert design.shape == (2, len(names))
    assert design.shape[1] >= 30
    assert pd.notna(design).all()
