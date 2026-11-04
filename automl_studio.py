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
    OPTIONAL_MODELS,
    detect_task,
    export_pipeline_bytes, 
    fetch_klines,
    market_to_dataset, 
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
    "🚀 Model Market Prediction Dashboard",
    ],)
st.sidebar.caption("Build, preprocess, train, and track high-performance Machine Learning models seamlessly.")


# ------------------------------------------------------------------
# LIVE MARKETdelta_color="off")
# ------------------------------------------------------------------
@st.cache_data(ttl=60)
def live(interval, limit):
    return fetch_klines("XRPUSDT", interval, limit=limit)


if page == "🔮 Live Market":
    st.title("🔮 XRP/USDT — Live Market")
    c1, c2 = st.columns(2)
    interval = c1.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    limit = c2.slider("Candles", 50, 1000, 300, 50)
    if st.button("🔄 Refresh"):
        live.clear()
    try:
        df = live(interval, limit)
        last, prev = df.iloc[-1], df.iloc[-2]
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Last price (USDT)",
            f"{last['close']:.5f}",
            f"{(last['close'] / prev['close'] - 1):.2%}")
        m2.metric(
            "Volume",
            f"{last['volume']:,.0f}")
        m3.metric(
            "Candles", len(df))
        fig = go.Figure(go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]))
        fig.update_layout(
            height=480,
            xaxis_rangeslider_visible=False,
            xaxis_title="Time (UTC)",
            yaxis_title="Price (USDT)")
        st.plotly_chart(fig,use_container_width=True)
        st.bar_chart(df.set_index("time")["volume"])
    except requests.RequestException as exc:
        st.error(f"Could not reach Binance: {exc}")
    st.info("Go to **Model Dashboard** and choose *Live market* as the data source to train on this data.")


# ------------------------------------------------------------------
# MODEL DASHBOARD
# ------------------------------------------------------------------
else:
    st.title("🚀 Professional Model Market Prediction Dashboard")
    st.caption("Upload data or use the live market → pick a target → get a champion model.")

    # ---- 1. DATA SOURCE ----
    st.header("1 · Data")
    source = st.radio("Source", ["Upload CSV", "🔮 Live market (XRP/USDT)"], horizontal=True)
    df, default_target, time_ordered = None, None, False

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
                st.session_state["market_ds"] = (ds, tgt)
            except requests.RequestException as exc:
                st.error(f"Binance download failed: {exc}")
        if "market_ds" in st.session_state:
            df, default_target = st.session_state["market_ds"]
            time_ordered = True

    if df is None:
        st.info("Provide a dataset to continue.")
        st.stop()

    st.write(f"**{len(df):,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # ---- 2. TARGET + SETTINGS ----
    st.header("2 · Target & settings")
    cols = df.columns.tolist()
    target = st.selectbox(
        "Target column", 
        cols,
        index=cols.index(default_target)
        if default_target in cols else len(cols) - 1)
    auto_task = detect_task(df[target])if df[target].notna().any() else "regression"
    st.success(f"Detected task: **{auto_task.upper()}**")
    with st.expander("Advanced"):
        override = st.selectbox(
            "Task",[
            "auto",
            "classification",
            "regression",
            ])
        time_ordered = st.checkbox(
            "Data is a time series (chronological split, no shuffling)",
            value=time_ordered)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("CV folds", 3, 10, 5)
        extra_models = st.multiselect(
            "Optional extra models (added to Linear / RandomForest / HistGradientBoosting)",
            ["XGBoost", "LightGBM"],
            help="Slower to train, sometimes more accurate. They compete fairly: "
                 "the champion is still chosen by cross-validation.")
        missing = [m for m in extra_models if not OPTIONAL_MODELS.get(m, False)]
        if missing:
            st.warning(f"Not installed: {', '.join(missing)}. "
                       f"Run `pip install {' '.join(m.lower() for m in missing)}` and restart the app.")
    st.header("🔮 Predict with a trained champion")
    select_pct = st.slider("Keep top % of features (feature selection)", 20, 100, 80, 10)
    out_factor = st.slider("Outlier fence (× IQR)", 1.5, 5.0, 3.0, 0.5)
    task = auto_task if override == "auto" else override

    # ---- 3. TRAIN ----
    if st.button("🚀 Run AutoML", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            res = run_automl(
                df, target, task, time_ordered, test_size, cv_folds, select_pct, out_factor,
                progress=lambda p, t: bar.progress(min(p, 1.0), text=t),
                extra_models=extra_models)
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
        st.caption("Missing values are imputed, outliers are clipped, numeric features are scaled, categories "
                   "are one-hot encoded and weak features are removed inside the pipeline (learned on training data only).")

    st.subheader(f"🏆 Champion: {res['champion_name']}  ({res['task']})")
    st.caption(f"Selected by cross-validation on the training set ({res['n_train']:,} rows). "
               f"Test metrics use {res['n_test']:,} unseen rows.")
    mcols = st.columns(len(res["champion_test_metrics"]))
    for col, (k, v) in zip(mcols, res["champion_test_metrics"].items()):
        base = res["baseline"].get(k)
        col.metric(f"Test {k}", f"{v:.4f}", f"baseline {base:.4f}" if base is not None else None,
            delta_color="off")

    if res.get("skipped_models"):
        st.warning(f"Skipped (package not installed): {', '.join(res['skipped_models'])}")
    st.dataframe(res["leaderboard"].round(5), use_container_width=True, hide_index=True)
    if res["task"] == "regression" and res["champion_test_metrics"]["R2"] <= 0.02:
        st.warning(
            "R² is near zero: the model is no better than predicting the average. "
            "This is common for price-return prediction — don't treat it as a trading signal.")

    # Correlation matrix
    st.subheader("Correlation matrix")
    num = res["clean_df"].select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        corr = num.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=max(400, 32 * len(corr)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for a correlation matrix.")

    # Confusion matrix or scatter
    if res["task"] == "classification":
        st.subheader("Confusion matrix (test set)")
        fig = px.imshow(res["confusion"], x=res["labels"], y=res["labels"], text_auto=True,
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("Actual vs predicted (test set)")
        y_t, y_p = res["y_test"], res["y_pred"]
        fig = px.scatter(x=y_t, y=y_p, labels={"x": "Actual", "y": "Predicted"}, opacity=0.6)
        lo, hi = float(min(y_t.min(), y_p.min())), float(max(y_t.max(), y_p.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

    # ---- 5. EXPORT ----
    st.header("4 · Export")
    st.download_button("⬇️ Download pipeline (.pkl)", export_pipeline_bytes(res),
                       file_name="champion_pipeline.pkl", mime="application/octet-stream")
    st.code(
        "import joblib, pandas as pd\n"
        "bundle = joblib.load('champion_pipeline.pkl')   # keep automl_engine.py next to your script\n"
        "pred = bundle['pipeline'].predict(new_df[bundle['feature_columns']])",
        language="python",
    )
