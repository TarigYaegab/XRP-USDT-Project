"""AutoML Studio — upload a CSV (or pull live XRP/USDT) and get a champion model as .pkl.

Run:  streamlit run automl_studio.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from automl_engine import (
    MARKET_FEATURES,
    OPTIONAL_MODELS,
    compute_metrics,
    detect_task,
    directional_accuracy,
    export_pipeline_bytes,
    fetch_klines,
    is_datetime_like,
    latest_market_features,
    load_bundle,
    make_bundle,
    market_backtest,
    market_to_dataset,
    model_card_markdown,
    run_automl,
)

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide")
st.sidebar.title(
    "🤖 XRP/USDT Live Price Prediction")
page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Live Market",
        "🧠 Model Dashboard",
        "🎯 Predict",
    ],)
st.sidebar.caption("Market data: Binance • Models: scikit-learn")

INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Streamlit >= 1.50 uses width="stretch"; `use_container_width` is deprecated there.
# Older versions only know `use_container_width`.
_ST_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
FULL = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}


# ------------------------------------------------------------------
# LIVE MARKET
# ------------------------------------------------------------------
@st.cache_data(ttl=60)
def live(interval, limit):
    return fetch_klines("XRPUSDT", interval, limit=limit)


if page == "🔮 Live Market":
    st.title("🔮 XRP/USDT — Live Market")
    c1, c2 = st.columns(2)
    interval = c1.selectbox("Interval", INTERVALS, index=3)
    limit = c2.slider("Candles", 50, 1000, 300, 50)
    if st.button("🔄 Refresh"):
        live.clear()
    try:
        df = live(interval, limit)
        last, prev = df.iloc[-1], df.iloc[-2]
        m1, m2, m3 = st.columns(3)
        m1.metric("Last price (USDT)", f"{last['close']:.5f}",
                  f"{(last['close'] / prev['close'] - 1):.2%}")
        m2.metric("Volume", f"{last['volume']:,.0f}")
        m3.metric("Candles", len(df))
        fig = go.Figure(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                       low=df["low"], close=df["close"]))
        fig.update_layout(height=480, xaxis_rangeslider_visible=False,
                          xaxis_title="Time (UTC)", yaxis_title="Price (USDT)")
        st.plotly_chart(fig, **FULL)
        st.bar_chart(df.set_index("time")["volume"])
    except requests.RequestException as exc:
        st.error(f"Could not reach Binance: {exc}")
    st.info("Go to **Model Dashboard** and choose *Live market* as the data source to train on this data.")


# ------------------------------------------------------------------
# MODEL DASHBOARD
# ------------------------------------------------------------------
elif page == "🧠 Model Dashboard":
    st.title("Model Market Prediction Dashboard")
    st.caption("Upload data or use the live market → pick a target → get a champion model.")

    # ---- 1. DATA SOURCE ----
    st.header("1 · Data")
    source = st.radio("Source", ["Upload CSV", "🔮 Live market (XRP/USDT)"], horizontal=True)
    df, default_target, time_ordered, market_meta = None, None, False, None

    if source == "Upload CSV":
        up = st.file_uploader("Upload a .csv file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
    else:
        a, b, c, d = st.columns(4)
        interval = a.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
        days = b.selectbox("History (days)", [30, 90, 180], index=2)
        horizon = c.number_input("Predict N candles ahead", 1, 72, 12)
        mtask = d.selectbox(
            "Market task",
            [
                "regression",
                "classification"
            ],
            format_func=lambda x: "Future return (regression)"
            if x == "regression"
            else "UP / DOWN / NEUTRAL (classification)")
        thr = st.number_input("Neutral threshold (%)", 0.0, 10.0, 0.8, 0.1) / 100 \
            if mtask == "classification" else 0.008
        if st.button("🔮 Fetch & prepare live data from Binance Market", type="primary"):
            try:
                with st.spinner("🔮 Importing Live candles from Binance..."):
                    candles = fetch_klines("XRPUSDT", interval, days=days)
                    ds, tgt = market_to_dataset(candles, int(horizon), mtask, thr)
                meta = {"source": "binance", "symbol": "XRPUSDT", "interval": interval,
                        "horizon": int(horizon), "market_task": mtask}
                st.session_state["market_ds"] = (ds, tgt, meta)
            except requests.RequestException as exc:
                st.error(f"Binance download failed: {exc}")
            except Exception as exc:
                st.error(f"Could not prepare the market data: {exc}")
        if "market_ds" in st.session_state:
            df, default_target, market_meta = st.session_state["market_ds"]
            time_ordered = True

    if df is None:
        st.info("Provide a dataset to continue.")
        st.stop()

    st.write(f"**{len(df):,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(10), **FULL, hide_index=True)

    # ---- 2. TARGET + SETTINGS ----
    st.header("2 · Target & settings")
    cols = df.columns.tolist()
    target = st.selectbox(
        "Target column",
        cols,
        index=cols.index(default_target)
        if default_target in cols else len(cols) - 1)
    if is_datetime_like(df[target]):
        st.error(f"'{target}' is a date/time column. Choose a numeric or category column as the target "
                 "(for example a price or a future return), not the time itself.")
        st.stop()
    auto_task = detect_task(df[target]) if df[target].notna().any() else "regression"
    st.success(f"Detected task: **{auto_task.upper()}**")
    gap_default = int(market_meta["horizon"]) if market_meta else 0
    with st.expander("Advanced"):
        override = st.selectbox(
            "Task", [
                "auto",
                "classification",
                "regression",
            ])
        time_ordered = st.checkbox(
            "Data is a time series (chronological split, no shuffling)",
            value=time_ordered)
        gap = st.number_input(
            "Purge gap (rows between train and test)", 0, 500, gap_default,
            help="Time series only. A target that looks N rows ahead overlaps the next N rows, "
                 "so those rows are skipped between train and test (and between CV folds) to avoid leakage. "
                 "Live-market data sets this to the prediction horizon automatically.")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("CV folds", 3, 10, 5)
        select_pct = st.slider(
            "Keep top % of features (100 = keep all)", 20, 100, 100, 10,
            help="Below 100 uses a linear F-test, which can drop features that tree models "
                 "would have used. Keep 100 unless you have many noisy columns.")
        out_factor = st.slider("Outlier fence (× IQR)", 1.5, 5.0, 3.0, 0.5)
        extra_models = st.multiselect(
            "Optional extra models (added to Linear / RandomForest / HistGradientBoosting)",
            ["XGBoost", "LightGBM"],
            help="Slower to train, sometimes more accurate. They compete fairly: "
                 "the champion is still chosen by cross-validation.")
        missing_pkgs = [m for m in extra_models if not OPTIONAL_MODELS.get(m, False)]
        if missing_pkgs:
            st.warning(f"Not installed: {', '.join(missing_pkgs)}. "
                       f"Run `pip install {' '.join(m.lower() for m in missing_pkgs)}` and restart the app.")
        tune = st.checkbox(
            "Tune the champion's hyperparameters (slower)",
            help="Random search on the training set only. The tuned model replaces the champion "
                 "only if its cross-validation score is better.")
        tune_iter = st.slider("Tuning trials", 5, 40, 15, 5) if tune else 15
    task = auto_task if override == "auto" else override

    # ---- 3. TRAIN ----
    if st.button("🚀 Run AutoML", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            res = run_automl(
                df, target, task, time_ordered, test_size, cv_folds, select_pct, out_factor,
                progress=lambda p, t: bar.progress(min(p, 1.0), text=t),
                extra_models=extra_models, gap=gap, tune=tune, tune_iter=tune_iter)
            res["meta"] = dict(market_meta) if market_meta else {"source": "upload"}
            st.session_state["result"] = res
        except Exception as exc:
            st.error(f"AutoML failed: {exc}")
        bar.empty()

    res = st.session_state.get("result")
    if res is None:
        st.stop()

    # ---- 4. RESULTS ----
    st.header("3 · Results")
    rep = res["report"]
    with st.expander("🧹 Cleaning report", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duplicates removed", rep["duplicates_removed"])
        c2.metric("Rows w/o target removed", rep["rows_missing_target_removed"])
        c3.metric("Columns dropped", len(rep["columns_dropped"]))
        c4.metric("Rows used", rep["rows_out"])
        if rep["columns_dropped"]:
            st.write(rep["columns_dropped"])
        if rep.get("datetime_columns"):
            st.write(f"Date/time columns expanded to hour / weekday / day / month: "
                     f"{', '.join(rep['datetime_columns'])}")
        if rep.get("sorted_by"):
            st.write(f"Rows sorted chronologically by: {rep['sorted_by']}")
        if res.get("gap"):
            st.write(f"Purge gap between train and test: {res['gap']} rows")
        st.caption("Missing values are imputed, outliers are clipped, numeric features are scaled, categories "
                   "are one-hot encoded and weak features are removed inside the pipeline (learned on training data only).")

    tuned = bool(res.get("tuning") and res["tuning"]["adopted"])
    st.subheader(f"🏆 Champion: {res['champion_name']}{' (tuned)' if tuned else ''}  ({res['task']})")
    st.caption(f"Selected by cross-validation on the training set ({res['n_train']:,} rows). "
               f"Test metrics use {res['n_test']:,} unseen rows.")
    mcols = st.columns(len(res["champion_test_metrics"]))
    for col, (k, v) in zip(mcols, res["champion_test_metrics"].items()):
        base = res["baseline"].get(k)
        col.metric(f"Test {k}", f"{v:.4f}", f"baseline {base:.4f}" if base is not None else None,
                   delta_color="off")

    if res.get("skipped_models"):
        st.warning(f"Skipped (package not installed): {', '.join(res['skipped_models'])}")
    st.dataframe(res["leaderboard"].round(5), **FULL, hide_index=True)
    if res["task"] == "regression" and res["champion_test_metrics"]["R2"] <= 0.02:
        st.warning(
            "R² is near zero: the model is no better than predicting the average. "
            "This is common for price-return prediction — don't treat it as a trading signal.")

    if res.get("tuning"):
        tun = res["tuning"]
        with st.expander("🎛 Hyperparameter tuning", expanded=False):
            st.write(f"{tun['metric']} before: **{tun['cv_before']:.4f}** → after: **{tun['cv_after']:.4f}** "
                     f"({'tuned model adopted' if tun['adopted'] else 'no improvement, default kept'}; "
                     f"{tun['n_iter']} trials)")
            st.json(tun["best_params"])

    # Feature importance
    imp = res.get("importance")
    if imp is not None and len(imp):
        st.subheader("Feature importance (permutation, test set)")
        top = imp.head(15).iloc[::-1]
        fig = px.bar(top, x="importance", y="feature", orientation="h", error_x="std",
                     labels={"importance": "Score drop when the feature is shuffled", "feature": ""})
        fig.update_layout(height=max(300, 28 * len(top) + 80))
        st.plotly_chart(fig, **FULL)
        st.caption("Higher = the model relies on it more. Values near zero mean the feature did not help "
                   "on unseen data. Correlated features can share importance.")

    # Correlation matrix
    st.subheader("Correlation matrix")
    num = res["clean_df"].select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        corr = num.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=max(400, 32 * len(corr)))
        st.plotly_chart(fig, **FULL)
    else:
        st.info("Need at least two numeric columns for a correlation matrix.")

    # Confusion matrix or scatter
    if res["task"] == "classification":
        st.subheader("Confusion matrix (test set)")
        fig = px.imshow(res["confusion"], x=res["labels"], y=res["labels"], text_auto=True,
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig, **FULL)
    else:
        st.subheader("Actual vs predicted (test set)")
        y_t, y_p = res["y_test"], res["y_pred"]
        fig = px.scatter(x=y_t, y=y_p, labels={"x": "Actual", "y": "Predicted"}, opacity=0.6)
        lo, hi = float(min(y_t.min(), y_p.min())), float(max(y_t.max(), y_p.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(dash="dash")))
        st.plotly_chart(fig, **FULL)

    # Market evaluation (future-return models only)
    meta = res.get("meta", {})
    if meta.get("source") == "binance" and meta.get("market_task") == "regression" and res["task"] == "regression":
        st.subheader("📈 Market evaluation (test period)")
        y_t, y_p = res["y_test"], res["y_pred"]
        da = directional_accuracy(y_t, y_p)
        share_up = float(np.mean(np.asarray(y_t) > 0))
        d1, d2 = st.columns(2)
        d1.metric("Directional accuracy", f"{da:.1%}",
                  f"always-guess-majority baseline {max(share_up, 1 - share_up):.1%}", delta_color="off")
        fee = st.slider("Trading fee per side (%)", 0.0, 0.5, 0.1, 0.01) / 100
        bt = market_backtest(y_t, y_p, meta["horizon"], fee)
        d2.metric("Trades taken", f"{bt['n_trades']} of {bt['n_periods']} periods")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Strategy return", f"{bt['strategy_return']:.2%}")
        e2.metric("Buy & hold return", f"{bt['buy_hold_return']:.2%}")
        e3.metric("Win rate", "n/a" if np.isnan(bt["win_rate"]) else f"{bt['win_rate']:.1%}")
        e4.metric("Max drawdown", f"{bt['max_drawdown']:.2%}")
        fig = px.line(bt["equity"], x="period", y=["Strategy", "Buy & hold"],
                      labels={"value": "Equity (start = 1)", "period": f"Non-overlapping {meta['horizon']}-candle periods"})
        st.plotly_chart(fig, **FULL)
        st.caption("Long-or-flat strategy: hold for N candles when the predicted return is above zero. "
                   "Only non-overlapping periods are used, so the sample is small and noisy. "
                   "Not a trading signal and not financial advice.")
        if bt["n_periods"] < 30:
            st.warning(f"Only {bt['n_periods']} periods in the test set: treat these numbers as anecdotal.")

    # ---- 5. EXPORT ----
    st.header("4 · Export")
    x1, x2, x3 = st.columns(3)
    x1.download_button("⬇️ Download pipeline (.pkl)", export_pipeline_bytes(res),
                       file_name="champion_pipeline.pkl", mime="application/octet-stream")
    x2.download_button("⬇️ Leaderboard (.csv)", res["leaderboard"].to_csv(index=False).encode(),
                       file_name="leaderboard.csv", mime="text/csv")
    x3.download_button("⬇️ Model card (.md)", model_card_markdown(res).encode(),
                       file_name="model_card.md", mime="text/markdown")
    st.code(
        "import joblib, pandas as pd\n"
        "bundle = joblib.load('champion_pipeline.pkl')   # keep automl_engine.py next to your script\n"
        "pred = bundle['pipeline'].predict(new_df[bundle['feature_columns']])",
        language="python",
    )
    st.info("Or use the **🎯 Predict** page to apply this model to the latest live candles or a new CSV.")


# ------------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------------
else:
    st.title("🎯 Predict with a trained champion")
    st.caption("Use the model you just trained, or load a saved .pkl, then predict on the latest live "
               "candles or on a new CSV.")

    model_options = []
    if st.session_state.get("result") is not None:
        model_options.append("Champion trained in this session")
    model_options.append("Upload a .pkl file")
    model_src = st.radio("Model", model_options, horizontal=True)

    bundle = None
    if model_src.startswith("Champion"):
        bundle = make_bundle(st.session_state["result"])
    else:
        st.warning("Only load .pkl files you created or trust: a pickle file can run code when it is loaded.")
        pkl = st.file_uploader("Upload champion_pipeline.pkl", type=["pkl", "joblib"])
        if pkl is not None:
            try:
                bundle = load_bundle(pkl.getvalue())
            except Exception as exc:
                st.error(f"Could not load the model: {exc}")
    if bundle is None:
        st.info("Select or upload a model to continue.")
        st.stop()

    pipe, needed, meta = bundle["pipeline"], bundle["feature_columns"], bundle.get("meta", {})
    i1, i2, i3 = st.columns(3)
    i1.metric("Model", bundle["champion"])
    i2.metric("Task", bundle["task"])
    i3.metric("Target", bundle["target"])
    with st.expander("Features the model needs"):
        st.write(needed)

    data_src = st.radio("Data to predict on", ["Latest live candles (XRP/USDT)", "Upload CSV with new data"],
                        horizontal=True)

    if data_src.startswith("Latest"):
        if not set(needed) <= set(MARKET_FEATURES):
            st.warning("This model was not trained on the live-market features, so it cannot predict from "
                       "live candles. Upload a CSV instead.")
            st.stop()
        trained_iv = meta.get("interval", "1h")
        interval = st.selectbox("Candle interval", INTERVALS,
                                index=INTERVALS.index(trained_iv) if trained_iv in INTERVALS else 3)
        if meta.get("interval") and interval != meta["interval"]:
            st.warning(f"The model was trained on {meta['interval']} candles; predictions on {interval} "
                       "candles may be unreliable.")
        if st.button("🔮 Fetch latest candles & predict", type="primary"):
            try:
                candles = fetch_klines("XRPUSDT", interval, limit=200)
            except requests.RequestException as exc:
                st.error(f"Could not reach Binance: {exc}")
                st.stop()
            feats = latest_market_features(candles, n=5)
            if feats.empty:
                st.error("Not enough candles to compute the features.")
                st.stop()
            preds = pipe.predict(feats[needed])
            out = pd.DataFrame({"time (UTC)": feats["time"], "close": feats["close"], "prediction": preds})
            last = out.iloc[-1]
            st.subheader(f"Latest closed candle: {last['time (UTC)']}")
            if bundle["task"] == "regression":
                if bundle["target"] == "target_future_return":
                    h = meta.get("horizon", "N")
                    p1, p2 = st.columns(2)
                    p1.metric(f"Predicted return over the next {h} candles", f"{last['prediction']:.3%}")
                    p2.metric("Implied price (USDT)", f"{last['close'] * (1 + last['prediction']):.5f}",
                              f"last close {last['close']:.5f}", delta_color="off")
                else:
                    st.metric(f"Predicted {bundle['target']}", f"{last['prediction']:.5f}")
            else:
                st.metric("Predicted class", str(last["prediction"]))
                try:
                    proba = pipe.predict_proba(feats[needed])[-1]
                    st.write({str(c): f"{p:.1%}" for c, p in zip(pipe.classes_, proba)})
                except Exception:
                    pass
            st.dataframe(out, **FULL, hide_index=True)
            st.caption("Not financial advice. Short-term price prediction is close to random; "
                       "check the model's market evaluation on the dashboard first.")
    else:
        csv = st.file_uploader("CSV with the model's feature columns", type=["csv"], key="pred_csv")
        if csv is not None:
            new_df = pd.read_csv(csv)
            new_df.columns = [str(c).strip() for c in new_df.columns]
            missing = [c for c in needed if c not in new_df.columns]
            if missing:
                st.error(f"The file is missing these columns: {missing}")
                st.stop()
            try:
                preds = pipe.predict(new_df[needed])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()
            out = new_df.copy()
            out[f"predicted_{bundle['target']}"] = preds
            st.success(f"Predicted {len(out):,} rows.")
            st.dataframe(out.head(100), **FULL, hide_index=True)
            st.download_button("⬇️ Download predictions (.csv)", out.to_csv(index=False).encode(),
                               file_name="predictions.csv", mime="text/csv")
            if bundle["target"] in new_df.columns:
                try:
                    m = compute_metrics(bundle["task"], new_df[bundle["target"]], preds)
                    st.subheader("Metrics against the target column in your file")
                    for col, (k, v) in zip(st.columns(len(m)), m.items()):
                        col.metric(k, f"{v:.4f}")
                except Exception:
                    pass
