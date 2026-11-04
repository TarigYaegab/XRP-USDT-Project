"""Generic AutoML engine: task detection, cleaning, preprocessing, model selection, export.

The exported pipeline (preprocessing + champion model) is a single sklearn Pipeline,
so loading the .pkl and calling .predict(raw_dataframe) just works.
NOTE: unpickling requires this module (automl_engine.py) to be importable,
because the pipeline contains the custom `Winsorizer` transformer.
"""
import io
import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Optional boosting libraries: the app still works if they are not installed.
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

OPTIONAL_MODELS = {"XGBoost": HAS_XGBOOST, "LightGBM": HAS_LIGHTGBM}


# ------------------------------------------------------------------
# 1. Task detection
# ------------------------------------------------------------------
def detect_task(y: pd.Series) -> str:
    """Return 'classification' or 'regression' from the target column."""
    y = y.dropna()
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    is_integer_like = np.allclose(y, np.round(y))
    if is_integer_like and (n_unique <= 10 or n_unique / max(len(y), 1) < 0.02):
        return "classification"
    return "regression"


# ------------------------------------------------------------------
# 2. Cleaning
# ------------------------------------------------------------------
def clean_data(df: pd.DataFrame, target: str, max_missing=0.6):
    """Drop duplicates / bad rows / junk columns. Returns (clean_df, report)."""
    report = {"rows_in": len(df), "cols_in": df.shape[1]}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    before = len(df)
    df = df.dropna(subset=[target])
    report["rows_missing_target_removed"] = before - len(df)

    dropped = {}
    for col in [c for c in df.columns if c != target]:
        s = df[col]
        if s.isna().mean() > max_missing:
            dropped[col] = f"more than {max_missing:.0%} missing"
        elif s.nunique(dropna=True) <= 1:
            dropped[col] = "constant column"
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique() / max(len(s), 1) > 0.9:
            dropped[col] = "ID-like / free text (almost all unique)"
    df = df.drop(columns=list(dropped))
    report["columns_dropped"] = dropped

    # Try converting numeric-looking text columns
    for col in df.columns:
        if col != target and not pd.api.types.is_numeric_dtype(df[col]):
            conv = pd.to_numeric(df[col], errors="coerce")
            nm = df[col].notna().sum()
            if nm and conv.notna().sum() / nm >= 0.95:
                df[col] = conv
    report["missing_cells_remaining"] = int(df.isna().sum().sum())
    report["rows_out"] = len(df)
    report["cols_out"] = df.shape[1]
    return df.reset_index(drop=True), report


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features to IQR fences learned on the training data only."""

    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, names=None):
        return names


# ------------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame, task: str, select_percent=80, outlier_factor=3.0):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("outliers", Winsorizer(factor=outlier_factor)),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)),
            ]),
            cat_cols,
        ))
    score_func = f_classif if task == "classification" else f_regression
    return Pipeline([
        ("columns", ColumnTransformer(transformers)),
        ("variance", VarianceThreshold(0.0)),
        ("select", SelectPercentile(score_func, percentile=int(select_percent))),
    ])


# ------------------------------------------------------------------
# 4. Models
# ------------------------------------------------------------------
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Lets classifiers that need integer labels (XGBoost) accept text labels like 'UP'/'DOWN'.

    Fits a LabelEncoder on y, trains the wrapped model on integers, and converts
    predictions back to the original labels, so the rest of the app is unchanged.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def get_models(task: str, extra_models=None):
    """Core models always run. `extra_models` may contain 'XGBoost' and/or 'LightGBM'
    (only added if the package is installed)."""
    extra_models = extra_models or []
    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05,
                max_leaf_nodes=15, l2_regularization=1.0,
                random_state=42),
        }
        if "XGBoost" in extra_models and HAS_XGBOOST:
            models["XGBoost"] = LabelEncodedClassifier(XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0))
        if "LightGBM" in extra_models and HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1)
        return models

    models = {
        "Ridge (Linear)": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=1.0,
            random_state=42),
    }
    if "XGBoost" in extra_models and HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
    if "LightGBM" in extra_models and HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    return models


def _metrics(task, y_true, y_pred):
    if task == "classification":
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


# ------------------------------------------------------------------
# 5. Full AutoML run
# ------------------------------------------------------------------
def run_automl(
    df: pd.DataFrame, target: str, task: str | None = None,
    time_ordered=False, test_size=0.2, cv_folds=5,
    select_percent=80, outlier_factor=3.0, progress=None, extra_models=None):
    """Clean -> preprocess -> train several models -> pick champion -> export-ready pipeline.

    extra_models: optional list like ['XGBoost', 'LightGBM'] added to the candidates.
    """
    extra_models = list(extra_models or [])
    skipped_models = [m for m in extra_models if not OPTIONAL_MODELS.get(m, False)]
    clean, report = clean_data(df, target)
    task = task or detect_task(clean[target])

    X = clean.drop(columns=[target])
    y = clean[target]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")
    if task == "classification" and y.value_counts().min() < 2:
        raise ValueError("Each class needs at least 2 rows. Pick another target or merge rare classes.")

    if time_ordered:
        cut = int(len(clean) * (1 - test_size))
        X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        strat = y if task == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                  random_state=42, stratify=strat)
        cv = (StratifiedKFold(cv_folds, shuffle=True, random_state=42)
              if task == "classification" else KFold(cv_folds, shuffle=True, random_state=42))

    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"

    def make_pipe(model):
        return Pipeline([("prep", build_preprocessor(X, task, select_percent, outlier_factor)),
                         ("model", model)])

    # Baseline for honest comparison
    from sklearn.dummy import DummyClassifier, DummyRegressor
    base = make_pipe(DummyClassifier(strategy="most_frequent") if task == "classification"
                     else DummyRegressor(strategy="mean")).fit(X_tr, y_tr)
    baseline = _metrics(task, y_te, base.predict(X_te))

    rows, fitted = [], {}
    models = get_models(task, extra_models)
    for i, (name, model) in enumerate(models.items()):
        if progress:
            progress(i / len(models), f"Training {name}...")
        pipe = make_pipe(model)
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        cv_value = cv_scores.mean() if task == "classification" else -cv_scores.mean()
        pipe.fit(X_tr, y_tr)
        test_m = _metrics(task, y_te, pipe.predict(X_te))
        rows.append({"Model": name,
                     ("CV_F1_Macro" if task == "classification" else "CV_RMSE"): cv_value,
                     **{f"Test_{k}": v for k, v in test_m.items()}})
        fitted[name] = pipe

    cv_col = "CV_F1_Macro" if task == "classification" else "CV_RMSE"
    leaderboard = pd.DataFrame(rows).sort_values(cv_col, ascending=(task == "regression"))
    leaderboard = leaderboard.reset_index(drop=True)
    champion_name = leaderboard.loc[0, "Model"]   # selected by CV on training data only

    champ_tr = fitted[champion_name]
    y_pred = champ_tr.predict(X_te)

    # Final export model: refit the champion on ALL data
    final_pipe = clone(champ_tr).fit(X, y)

    if progress:
        progress(1.0, "Done")

    out = {
        "task": task, "report": report, "leaderboard": leaderboard, "baseline": baseline,
        "champion_name": champion_name, "champion_test_metrics": _metrics(task, y_te, y_pred),
        "pipeline": final_pipe, "y_test": y_te.to_numpy(), "y_pred": y_pred,
        "clean_df": clean, "target": target, "feature_columns": X.columns.tolist(),
        "n_train": len(X_tr), "n_test": len(X_te),
        "skipped_models": skipped_models,
    }
    if task == "classification":
        labels = sorted(pd.unique(y))
        out["labels"] = [str(l) for l in labels]
        out["confusion"] = confusion_matrix(y_te, y_pred, labels=labels)
    return out


def export_pipeline_bytes(result) -> bytes:
    """Serialise the full pipeline (preprocessing + model) with metadata to .pkl bytes."""
    buf = io.BytesIO()
    joblib.dump({
        "pipeline": result["pipeline"],
        "task": result["task"],
        "target": result["target"],
        "feature_columns": result["feature_columns"],
        "champion": result["champion_name"],
    }, buf)
    return buf.getvalue()


# ------------------------------------------------------------------
# 6. Live market helpers (XRP/USDT from Binance)
# ------------------------------------------------------------------
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]


def fetch_klines(symbol="XRPUSDT", interval="1h", limit=500, days=None):
    """Latest `limit` candles, or `days` of history (paginated)."""
    interval_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
                   "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[interval]
    rows = []
    if days is None:
        r = requests.get(BINANCE_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
        r.raise_for_status()
        rows = r.json()
    else:
        end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        cursor = end - days * 86_400_000
        while cursor < end:
            r = requests.get(BINANCE_URL, params={"symbol": symbol, "interval": interval,
                                                  "startTime": cursor, "endTime": end, "limit": 1000}, timeout=20)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + interval_ms
            if len(batch) < 1000:
                break
    df = pd.DataFrame(rows, columns=_COLS)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates("open_time").sort_values("time").reset_index(drop=True)
    # drop the still-open candle
    step = pd.Timedelta(milliseconds=interval_ms)
    df = df[df["time"] + step <= pd.Timestamp.now(tz="UTC")]
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def market_to_dataset(candles: pd.DataFrame, horizon=12, task="regression", threshold=0.008):
    """Engineer features from OHLCV and add a future-return target (no leakage in features)."""
    df = candles.copy().reset_index(drop=True)
    c = df["close"]
    for n in (1, 3, 6, 12):
        df[f"Return_{n}"] = c.pct_change(n)
    df["MA_5_Ratio"] = c / c.rolling(5).mean() - 1
    df["MA_20_Ratio"] = c / c.rolling(20).mean() - 1
    df["Volatility_10"] = df["Return_1"].rolling(10).std()
    delta = c.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    for p in (7, 14):
        ag, al = gain.rolling(p).mean(), loss.rolling(p).mean()
        rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
        rsi = rsi.mask((al == 0) & (ag > 0), 100).mask((al == 0) & (ag == 0), 50)
        df[f"RSI_{p}"] = rsi
    rng = df["high"] - df["low"]
    df["Candle_Range"] = rng / c
    df["Body_Ratio"] = (c - df["open"]).abs() / rng.replace(0, np.nan)
    df["Volume_Change"] = df["volume"].pct_change()

    fut = c.shift(-horizon) / c - 1
    if task == "regression":
        df["target_future_return"] = fut
        target = "target_future_return"
    else:
        df["target_direction"] = np.where(fut > threshold, "UP", np.where(fut < -threshold, "DOWN", "NEUTRAL"))
        df.loc[fut.isna(), "target_direction"] = np.nan
        target = "target_direction"
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target] + [
        "Return_12", "MA_20_Ratio", "Volatility_10", "RSI_14", "Volume_Change"])
    df = df.drop(columns=["open", "high", "low", "close", "volume", "time"])  # raw price levels are non-stationary
    return df.reset_index(drop=True), target
