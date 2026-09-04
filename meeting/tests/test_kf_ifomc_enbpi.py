from types import SimpleNamespace

import numpy as np
import pandas as pd

from kf_forecasting.models.kf_ifomc_enbpi import (
    IFOMCIntervalConfig,
    rolling_ifomc_from_fitted,
)


def _fake_fitted():
    raw_residuals = np.array([-3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2], float)
    forecast = SimpleNamespace(
        truth=np.array([10.0, 12.0, 9.0]),
        point=np.array([10.5, 10.5, 10.5]),
        config=SimpleNamespace(alpha=0.05, oob_bias_correction=False),
    )
    return SimpleNamespace(
        forecast=forecast,
        crossfit_predictions=pd.DataFrame({"final_residual": raw_residuals}),
        selected_residual_window=None,
    )


def test_rolling_ifomc_updates_once_after_each_forecast():
    result = rolling_ifomc_from_fitted(
        _fake_fitted(),
        pd.RangeIndex(3),
        config=IFOMCIntervalConfig(n_states=3, min_source_transitions=1),
    )
    initial_transitions = result.state_table["initial_count"].sum() - 1
    final_transitions = result.state_table["final_count"].sum() - 1
    assert final_transitions - initial_transitions == 3
    assert len(result.predictions) == 3
    assert np.all(result.predictions["lower"] <= result.predictions["upper"])


def test_ifomc_keeps_the_fitted_m6_point_forecast():
    fitted = _fake_fitted()
    result = rolling_ifomc_from_fitted(
        fitted,
        pd.RangeIndex(3),
        config=IFOMCIntervalConfig(n_states=3, min_source_transitions=1),
    )
    np.testing.assert_allclose(result.predictions["point"], fitted.forecast.point)
