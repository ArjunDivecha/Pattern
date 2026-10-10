"""Guard the economic comparison, timing and separation of selection/accounting."""
import numpy as np
import pandas as pd
import pytest

from research.cost_aware.penalty_sweep import cohort, matured_sigma, pretrade_fields, select


def candidates():
    return pd.DataFrame({
        "security": ["A", "B", "C", "D"],
        "signal_strength": [4, 3, 2, 1],
        "alpha": [.02, .02, .02, .02],
        "pre_cost": [.03, .025, .005, .001],
        "pretrade_eligible": [True]*4,
    })


def test_zero_penalty_preserves_model_order_with_flat_calibration():
    assert select(candidates(), 0, 4, n=2).security.tolist() == ["A", "B"]
    assert select(candidates(), 1, 4, n=2).security.tolist() == ["D", "C"]


def test_rank_cap_limits_replacements():
    got = select(candidates(), 1, 3, n=2)
    assert got.security.tolist() == ["C", "B"]
    assert got.eligible_signal_rank.max() <= 3


def test_future_data_cannot_change_selection():
    x = candidates()
    x["absolute_forward_return"] = [np.nan, -.9, 10, .5]
    x["total_cost"] = [np.nan, 1, 0, 0]
    got = select(x, 0, 4, n=2).security.tolist()
    x["absolute_forward_return"] = 0
    x["total_cost"] = 0
    assert select(x, 0, 4, n=2).security.tolist() == got == ["A", "B"]


def portfolio():
    return pd.DataFrame({"leg": ["long", "short"],
                         "absolute_forward_return": [.04, -.02],
                         "fill_spread_rt": [.001, .002],
                         "fill_comm_reg_rt": [.001, .001],
                         "borrow_cost": [0, .003],
                         "impact_cost": [.002, .002],
                         "pre_cost": [.004, .008],
                         "total_cost": [.004, .008]})


def test_full_costs_charged_even_when_selection_penalty_is_zero():
    x = portfolio()
    x["penalty"] = 0
    first = cohort(x, n=1)
    x["penalty"] = 1
    second = cohort(x, n=1)
    assert first == second
    assert first["gross"] == pytest.approx(.06)
    assert first["net"] == pytest.approx(.048)
    assert first["net_no_impact"] == pytest.approx(.052)


def test_missing_cost_invalidates_whole_cohort():
    x = portfolio()
    x.loc[0, "total_cost"] = np.nan
    result = cohort(x, n=1)
    assert not result["complete"]
    assert np.isnan(result["net"])


def test_missing_future_dates_do_not_make_expected_short_borrow_free():
    x = pd.DataFrame({"leg": ["short"], "rank": [1],
                      "expected_relative_return": [-.02],
                      "pre_spread_rt": [.001], "pre_comm_reg_rt": [.002],
                      "pre_impact_unit": [0.0], "pre_fee_native": [.12],
                      "pre_adv20": [1e6], "scheduled_days": [6],
                      "fill_calendar_days": [0], "absolute_forward_return": [np.nan]})
    result = pretrade_fields(x).iloc[0]
    assert result.pre_cost == pytest.approx(.003 + 1.02 * .12 * 6 / 360)
    assert result.pretrade_eligible


def test_volatility_excludes_shifted_but_not_yet_finished_return():
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    x = pd.DataFrame({"security": "A", "date": dates,
                      "exit_date": dates + pd.Timedelta(days=3),
                      "absolute_forward_return": np.arange(15)/100})
    got = matured_sigma(x)
    # At day 13, outcomes formed days 1..10 are complete, while days 11/12 are not.
    row = got[got.date.eq(dates[12])].iloc[0]
    assert row.sigma_daily == pytest.approx(np.std(np.arange(10)/100, ddof=1)/np.sqrt(5))
    assert row.sigma_latest_exit == dates[12]
    assert pd.isna(got[got.date.eq(dates[11])].iloc[0].sigma_daily)
