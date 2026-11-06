"""Tests for automl_engine. Run:  pytest -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import automl_engine as ae  # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "cat": rng.choice(list("xyz"), n),
    })
    df["y"] = 2 * df["a"] - df["b"] + rng.normal(scale=0.3, size=n)
    df["label"] = np.where(df["y"] > 0.5, "hi", np.where(df["y"] < -0.5, "lo", "mid"))
    df.loc[rng.choice(n, 30, replace=False), "a"] = np.nan
    return df


@pytest.fixture(scope="module")
def candles():
    rng = np.random.default_rng(1)
    n = 700
    p = 0.5 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "open": p * 1.001, "high": p * 1.004, "low": p * 0.996, "close": p,
        "volume": rng.uniform(1e5, 2e5, n),
    })


# ------------------------------------------------------------------ detect_task
def test_detect_task_text_is_classification():
    assert ae.detect_task(pd.Series(["UP", "DOWN", "UP"])) == "classification"


def test_detect_task_few_integers_is_classification():
    assert ae.detect_task(pd.Series([0, 1, 1, 0, 1] * 20)) == "classification"


def test_detect_task_continuous_is_regression():
    assert ae.detect_task(pd.Series(np.random.default_rng(0).normal(size=200))) == "regression"


# ------------------------------------------------------------------ cleaning
def test_clean_data_report(table):
    base = table.copy()
    base["const"] = 1
    base["ident"] = [f"id{i}" for i in range(len(base))]
    df = pd.concat([base, base.head(10)])          # 10 exact duplicate rows
    clean, rep = ae.clean_data(df.drop(columns=["label"]), "y")
    assert rep["duplicates_removed"] == 10
    assert "const" in rep["columns_dropped"]
    assert "ident" in rep["columns_dropped"]
    assert rep["datetime_columns"] == ["time"]          # kept, not dropped as ID-like
    assert "time" in clean.columns


def test_numeric_text_is_converted():
    df = pd.DataFrame({"n": ["1.5", "2.5", "3.5", "4.5"] * 10, "y": np.arange(40.0)})
    clean, _ = ae.clean_data(df, "y")
    assert pd.api.types.is_numeric_dtype(clean["n"])


def test_is_datetime_like():
    assert ae.is_datetime_like(pd.Series(["2026-01-01 10:00:00", "2026-01-02 11:00:00"] * 5))
    assert not ae.is_datetime_like(pd.Series(["UP", "DOWN"] * 5))
    assert not ae.is_datetime_like(pd.Series(["10:30", "11:45"] * 5))
    assert not ae.is_datetime_like(pd.Series(["12345678", "87654321"] * 5))   # numbers as text
    assert not ae.is_datetime_like(pd.Series([1.0, 2.0, 3.0]))


# ------------------------------------------------------------------ run_automl
def test_regression_end_to_end(table):
    r = ae.run_automl(table.drop(columns=["label"]), "y")
    assert r["task"] == "regression"
    assert r["champion_test_metrics"]["R2"] > 0.8
    assert r["champion_test_metrics"]["RMSE"] < r["baseline"]["RMSE"]


def test_classification_end_to_end(table):
    r = ae.run_automl(table.drop(columns=["y"]), "label")
    assert r["task"] == "classification"
    assert r["champion_test_metrics"]["Accuracy"] > r["baseline"]["Accuracy"]
    assert r["confusion"].shape == (3, 3)


def test_datetime_target_is_rejected(table):
    with pytest.raises(ValueError, match="date/time"):
        ae.run_automl(table, "time")


def test_regression_on_text_target_is_rejected(table):
    with pytest.raises(ValueError, match="not numeric"):
        ae.run_automl(table.drop(columns=["y"]), "label", task="regression")


def test_exported_model_accepts_raw_dates(table):
    """The date column is expanded INSIDE the pipeline, so raw rows predict directly."""
    r = ae.run_automl(table.drop(columns=["label"]), "y")
    assert "dates" in dict(r["pipeline"].steps)
    bundle = ae.load_bundle(ae.export_pipeline_bytes(r))
    raw = table[bundle["feature_columns"]].head(5)
    assert raw["time"].dtype != "datetime64[ns]"          # still raw text
    pred = bundle["pipeline"].predict(raw)
    assert len(pred) == 5 and np.isfinite(pred).all()


def test_shuffled_rows_are_sorted_by_time(table):
    shuffled = table.drop(columns=["label"]).sample(frac=1, random_state=3)
    clean, rep = ae.clean_data(shuffled, "y", sort_by_time=True)
    assert rep["sorted_by"] == "time"
    times = pd.to_datetime(clean["time"])
    assert times.is_monotonic_increasing


# ------------------------------------------------------------------ time series gap
def test_purge_gap_shrinks_training_set(table):
    t = table.drop(columns=["label"])
    r0 = ae.run_automl(t, "y", time_ordered=True, gap=0, explain=False)
    r12 = ae.run_automl(t, "y", time_ordered=True, gap=12, explain=False)
    assert r0["n_train"] - r12["n_train"] == 12
    assert r0["n_test"] == r12["n_test"]
    assert r12["gap"] == 12


def test_gap_ignored_for_random_split(table):
    r = ae.run_automl(table.drop(columns=["label"]), "y", time_ordered=False, gap=50, explain=False)
    assert r["gap"] == 0


def test_gap_too_large_raises(table):
    with pytest.raises(ValueError, match="purge gap"):
        ae.run_automl(table.drop(columns=["label"]), "y", time_ordered=True, gap=350)


# ------------------------------------------------------------------ extras, tuning, importance
@pytest.mark.skipif(not (ae.HAS_XGBOOST and ae.HAS_LIGHTGBM), reason="xgboost/lightgbm not installed")
def test_extra_models_with_text_labels(table):
    r = ae.run_automl(table.drop(columns=["y"]), "label", extra_models=["XGBoost", "LightGBM"], explain=False)
    assert {"XGBoost", "LightGBM"} <= set(r["leaderboard"]["Model"])
    assert set(r["pipeline"].predict(table[r["feature_columns"]].head(20))) <= {"hi", "mid", "lo"}


def test_missing_optional_model_is_skipped(table, monkeypatch):
    monkeypatch.setitem(ae.OPTIONAL_MODELS, "XGBoost", False)
    monkeypatch.setattr(ae, "HAS_XGBOOST", False)
    r = ae.run_automl(table.drop(columns=["label"]), "y", extra_models=["XGBoost"], explain=False)
    assert r["skipped_models"] == ["XGBoost"]
    assert "XGBoost" not in set(r["leaderboard"]["Model"])


def test_tuning_only_adopted_when_better(table):
    r = ae.run_automl(table.drop(columns=["label"]), "y", tune=True, tune_iter=4, explain=False)
    t = r["tuning"]
    assert t["adopted"] == (t["cv_after"] < t["cv_before"])      # RMSE: lower is better
    assert set(t["best_params"])                                   # parameters were reported


def test_permutation_importance_finds_real_features(table):
    r = ae.run_automl(table.drop(columns=["label"]), "y")
    imp = r["importance"].set_index("feature")["importance"]
    assert imp["a"] > 0.5 and imp["b"] > 0.2                        # true drivers
    assert imp["a"] > imp["cat"]                                    # noise column matters less


# ------------------------------------------------------------------ market helpers
def test_market_features_have_no_lookahead(candles):
    """Features at row t must not change when later candles are appended."""
    full = ae.build_market_features(candles)
    part = ae.build_market_features(candles.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400][ae.MARKET_FEATURES], part[ae.MARKET_FEATURES])


def test_market_target_is_future_return(candles):
    ds, target = ae.market_to_dataset(candles, horizon=12, task="regression")
    feats = ae.build_market_features(candles).dropna(subset=ae._WARMUP_COLS)
    expected = (candles["close"].shift(-12) / candles["close"] - 1).loc[feats.index]
    first = feats.index[0]
    assert ds.loc[0, target] == pytest.approx(expected.loc[first])
    assert set(ds.columns) == set(ae.MARKET_FEATURES) | {target}      # raw prices removed


def test_market_classification_labels(candles):
    ds, target = ae.market_to_dataset(candles, horizon=12, task="classification", threshold=0.01)
    assert set(ds[target]) <= {"UP", "DOWN", "NEUTRAL"}


def test_latest_features_use_most_recent_candles(candles):
    latest = ae.latest_market_features(candles, n=3)
    assert len(latest) == 3
    assert latest["time"].iloc[-1] == candles["time"].iloc[-1]
    assert latest[ae.MARKET_FEATURES[:3]].notna().all().all()


def test_market_model_predicts_latest_candles(candles):
    ds, target = ae.market_to_dataset(candles, horizon=12, task="regression")
    r = ae.run_automl(ds, target, time_ordered=True, gap=12, explain=False)
    bundle = ae.load_bundle(ae.export_pipeline_bytes(r))
    latest = ae.latest_market_features(candles, n=2)
    pred = bundle["pipeline"].predict(latest[bundle["feature_columns"]])
    assert len(pred) == 2 and np.isfinite(pred).all()


# ------------------------------------------------------------------ evaluation helpers
def test_directional_accuracy():
    assert ae.directional_accuracy([1, -1, 1, -1], [2, -2, -1, 1]) == pytest.approx(0.5)
    assert ae.directional_accuracy([0, 0], [1, -1]) != ae.directional_accuracy([0, 0], [1, -1])  # NaN


def test_backtest_arithmetic():
    # horizon=1, perfect foresight, no fees: long only when the return is positive
    r = np.array([0.10, -0.05, 0.02])
    bt = ae.market_backtest(r, r, horizon=1, fee=0.0)
    assert bt["strategy_return"] == pytest.approx(1.10 * 1.0 * 1.02 - 1)
    assert bt["buy_hold_return"] == pytest.approx(1.10 * 0.95 * 1.02 - 1)
    assert bt["n_trades"] == 2


def test_backtest_fee_is_charged_per_trade():
    r = np.array([0.01, 0.01])
    bt = ae.market_backtest(r, r, horizon=1, fee=0.001)
    assert bt["strategy_return"] == pytest.approx((1 + 0.01 - 0.002) ** 2 - 1)


def test_backtest_uses_non_overlapping_periods():
    r = np.full(24, 0.001)
    bt = ae.market_backtest(r, r, horizon=12, fee=0.0)
    assert bt["n_periods"] == 2


# ------------------------------------------------------------------ bundle
def test_load_bundle_rejects_foreign_pickle():
    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump({"hello": 1}, buf)
    with pytest.raises(ValueError, match="not an AutoML Studio"):
        ae.load_bundle(buf.getvalue())


def test_model_card_mentions_key_facts(table):
    r = ae.run_automl(table.drop(columns=["label"]), "y", time_ordered=True, gap=5, explain=False)
    card = ae.model_card_markdown(r)
    assert r["champion_name"] in card and "`y`" in card and "purge gap 5" in card
