"""Generic AutoML engine: task detection, cleaning, preprocessing, model selection, export.

The exported pipeline (preprocessing + champion model) is a single sklearn Pipeline,
so loading the .pkl and calling .predict(raw_dataframe) just works.
NOTE: unpickling requires this module (automl_engine.py) to be importable,
because the pipeline contains custom transformers (`Winsorizer`, `DateTimeExpander`).
"""
import io
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import loguniform, randint, uniform

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
from sklearn.inspection import permutation_importance
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
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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
# 2. Date/time columns
# ------------------------------------------------------------------
def _parse_dates(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", utc=True)


def is_datetime_like(s: pd.Series) -> bool:
    """True for datetime dtype columns and for text columns that are (almost) all dates."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    nn = s.dropna()
    if nn.empty:
        return False
    nn = nn.astype(str)
    if nn.str.len().median() < 6:                       # "10:30", "x", "UP" are not dates
        return False
    if pd.to_numeric(nn, errors="coerce").notna().mean() >= 0.95:   # numbers stored as text
        return False
    sample = nn if len(nn) <= 2000 else nn.sample(2000, random_state=0)
    return bool(_parse_dates(sample).notna().mean() >= 0.95)


class DateTimeExpander(BaseEstimator, TransformerMixin):
    """Replace date/time columns by hour, day-of-week, day and month features.

    It lives inside the pipeline, so the exported .pkl accepts the raw date column
    (text or datetime) at prediction time.
    """

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns_:
            dt = _parse_dates(X[c])
            X[f"{c}_hour"] = dt.dt.hour
            X[f"{c}_dayofweek"] = dt.dt.dayofweek
            X[f"{c}_day"] = dt.dt.day
            X[f"{c}_month"] = dt.dt.month
        return X.drop(columns=self.columns_)


# ------------------------------------------------------------------
# 3. Cleaning
# ------------------------------------------------------------------
def clean_data(df: pd.DataFrame, target: str, max_missing=0.6, sort_by_time=False):
    """Drop duplicates / bad rows / junk columns. Returns (clean_df, report).

    Date/time columns are detected and kept (they are expanded later, inside the pipeline).
    If `sort_by_time` is True and a date column exists, rows are sorted chronologically.
    """
    target = str(target).strip()
    report = {"rows_in": len(df), "cols_in": df.shape[1]}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    dt_cols = [c for c in df.columns if c != target and is_datetime_like(df[c])]
    report["sorted_by"] = None
    if sort_by_time and dt_cols:
        key = _parse_dates(df[dt_cols[0]])
        df = df.assign(__k=key).sort_values("__k", kind="stable").drop(columns="__k")
        report["sorted_by"] = dt_cols[0]

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
        elif col in dt_cols:
            continue
        elif s.nunique(dropna=True) <= 1:
            dropped[col] = "constant column"
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique() / max(len(s), 1) > 0.9:
            dropped[col] = "ID-like / free text (almost all unique)"
    df = df.drop(columns=list(dropped))
    report["columns_dropped"] = dropped
    dt_cols = [c for c in dt_cols if c not in dropped]
    report["datetime_columns"] = dt_cols

    # Try converting numeric-looking text columns
    for col in df.columns:
        if col != target and col not in dt_cols and not pd.api.types.is_numeric_dtype(df[col]):
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
# 4. Preprocessing
# ------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame, task: str, select_percent=100, outlier_factor=3.0):
    """X is only used to read column names and dtypes. select_percent=100 keeps every feature."""
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
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01, sparse_output=False)),
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
# 5. Models
# ------------------------------------------------------------------
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Lets classifiers that need integer labels (XGBoost) accept text labels like 'UP'/'DOWN'.

    Fits a LabelEncoder on y, trains the wrapped model on integers, and converts
    predictions back to the original labels. With balance=True it also applies
    balanced sample weights (XGBoost has no class_weight option).
    """

    def __init__(self, estimator=None, balance=True):
        self.estimator = estimator
        self.balance = balance

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        y_enc = self.encoder_.transform(y)
        self.estimator_ = clone(self.estimator)
        if self.balance:
            self.estimator_.fit(X, y_enc, sample_weight=compute_sample_weight("balanced", y_enc))
        else:
            self.estimator_.fit(X, y_enc)
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


def _search_space(name: str, task: str) -> dict:
    """Random-search distributions for hyperparameter tuning of the champion."""
    pre = "model__estimator__" if (name == "XGBoost" and task == "classification") else "model__"
    model_space = {
        "LogisticRegression": {"C": loguniform(1e-3, 1e2)},
        "Ridge (Linear)": {"alpha": loguniform(1e-3, 1e3)},
        "RandomForest": {
            "n_estimators": randint(100, 400), "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", 0.5, 1.0]},
        "HistGradientBoosting": {
            "learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": randint(8, 64),
            "l2_regularization": loguniform(1e-3, 10), "max_iter": randint(100, 400)},
        "XGBoost": {
            "n_estimators": randint(100, 400), "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(2, 8), "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": loguniform(1e-2, 10)},
        "LightGBM": {
            "n_estimators": randint(100, 400), "learning_rate": loguniform(0.01, 0.3),
            "num_leaves": randint(8, 64), "min_child_samples": randint(5, 40),
            "reg_lambda": loguniform(1e-2, 10)},
    }.get(name, {})
    space = {pre + k: v for k, v in model_space.items()}
    space["prep__select__percentile"] = [50, 70, 85, 100]
    return space


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


compute_metrics = _metrics   # public alias


# ------------------------------------------------------------------
# 6. Full AutoML run
# ------------------------------------------------------------------
def _max_ts_folds(n_train: int, gap: int, wanted: int, min_first_train: int = 10) -> int:
    """Largest k <= wanted for which TimeSeriesSplit(k, gap) is feasible on n_train rows
    (and its first fold still has at least `min_first_train` training rows). 0 if none."""
    for k in range(int(wanted), 1, -1):
        if n_train - gap - (n_train // (k + 1)) * k >= min_first_train:
            return k
    return 0


def run_automl(
    df: pd.DataFrame, target: str, task: str | None = None,
    time_ordered=False, test_size=0.2, cv_folds=5,
    select_percent=100, outlier_factor=3.0, progress=None, extra_models=None,
    gap=0, tune=False, tune_iter=15, explain=True):
    """Clean -> preprocess -> train several models -> pick champion -> export-ready pipeline.

    extra_models: optional list like ['XGBoost', 'LightGBM'] added to the candidates.
    gap:(time series only) rows purged between train and test, and between CV folds.
    Use the prediction horizon: a target looking N rows ahead overlaps the next N rows.
    tune:random-search the champion's hyperparameters (adopted only if CV improves).
    explain:compute permutation importance on the test set.
    """
    target = str(target).strip()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    if is_datetime_like(df[target]):
        raise ValueError(
            f"Target '{target}' is a date/time column. Choose a numeric or category column "
            "(for example a price or a future return), not the time itself.")

    extra_models = list(extra_models or [])
    skipped_models = [m for m in extra_models if not OPTIONAL_MODELS.get(m, False)]
    gap = int(gap) if time_ordered else 0

    clean, report = clean_data(df, target, sort_by_time=time_ordered)
    if len(clean) < 30:
        raise ValueError(f"Only {len(clean)} usable rows after cleaning; at least 30 are needed.")
    task = task or detect_task(clean[target])
    if task == "regression" and not pd.api.types.is_numeric_dtype(clean[target]):
        raise ValueError("The target is not numeric, so regression is impossible. Use classification.")
    dt_cols = report["datetime_columns"]

    X = clean.drop(columns=[target])
    y = clean[target]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")
    if task == "classification" and y.value_counts().min() < 2:
        raise ValueError("Each class needs at least 2 rows. Pick another target or merge rare classes.")

    run_warnings = []
    if time_ordered:
        cut = int(len(clean) * (1 - test_size))
        train_end = cut - gap
        if train_end < 30:
            raise ValueError(
                f"Not enough rows for a chronological split with a purge gap of {gap}: "
                f"only {max(train_end, 0)} training rows. Use a longer history, a smaller test size "
                "or a smaller gap.")
        X_tr, X_te, y_tr, y_te = X.iloc[:train_end], X.iloc[cut:], y.iloc[:train_end], y.iloc[cut:]
        folds = _max_ts_folds(len(X_tr), gap, cv_folds)
        if folds == 0:
            raise ValueError(
                f"Only {len(X_tr)} training rows: too few for time-series cross-validation with a purge "
                f"gap of {gap}. Use a longer history (at least ~100 rows are recommended), a smaller "
                "gap, or a smaller test size.")
        if folds < cv_folds:
            run_warnings.append(
                f"CV folds reduced from {cv_folds} to {folds}: only {len(X_tr)} training rows "
                f"with a purge gap of {gap}.")
        cv = TimeSeriesSplit(n_splits=folds, gap=gap)
    else:
        strat = y if task == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size,random_state=42, stratify=strat)
        cv = (StratifiedKFold(cv_folds, shuffle=True, random_state=42)
            if task == "classification" else KFold(cv_folds, shuffle=True, random_state=42))

    if len(X_tr) < 200:
        run_warnings.append(
            f"Only {len(X_tr)} training rows: results will be very unstable. "
            "Use more data before trusting any metric.")

    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"

    # Column names/dtypes AFTER date expansion decide which columns are numeric or categorical.
    X_types = DateTimeExpander(columns=dt_cols).fit(X).transform(X) if dt_cols else X

    def make_pipe(model):
        steps = []
        if dt_cols:
            steps.append(("dates", DateTimeExpander(columns=dt_cols)))
        steps.append(("prep", build_preprocessor(X_types, task, select_percent, outlier_factor)))
        steps.append(("model", model))
        return Pipeline(steps)

    # Baseline for honest comparison
    from sklearn.dummy import DummyClassifier, DummyRegressor
    base = make_pipe(DummyClassifier(strategy="most_frequent") if task == "classification"
                     else DummyRegressor(strategy="mean")).fit(X_tr, y_tr)
    baseline = _metrics(task, y_te, base.predict(X_te))

    rows, fitted = [], {}
    models = get_models(task, extra_models)
    for i, (name, model) in enumerate(models.items()):
        if progress:
            progress(i / (len(models) + 1), f"Training {name}...")
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

    # Optional hyperparameter tuning of the champion (train-only CV; adopted only if better)
    tuning = None
    if tune:
        if progress:
            progress(len(models) / (len(models) + 1), f"Tuning {champion_name}...")
        cv_before = float(leaderboard.loc[0, cv_col])
        search = RandomizedSearchCV(
            make_pipe(get_models(task, extra_models)[champion_name]),
            _search_space(champion_name, task), n_iter=int(tune_iter), cv=cv,
            scoring=scoring, random_state=42, n_jobs=1, refit=True, error_score=np.nan)
        search.fit(X_tr, y_tr)
        cv_after = float(search.best_score_) if task == "classification" else float(-search.best_score_)
        improved = cv_after > cv_before if task == "classification" else cv_after < cv_before
        tuning = {
            "adopted": bool(improved), "cv_before": cv_before, "cv_after": cv_after,
            "metric": cv_col, "n_iter": int(tune_iter),
            "best_params": {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()},
        }
        if improved:
            champ_tr = search.best_estimator_

    y_pred = champ_tr.predict(X_te)

    # Final export model: refit the champion (with tuned params, if adopted) on ALL data
    final_pipe = clone(champ_tr).fit(X, y)

    # Permutation importance on the unseen test set (raw columns, so dates count as one feature)
    importance = None
    if explain:
        try:
            Xs = X_te if len(X_te) <= 1500 else X_te.sample(1500, random_state=0)
            pi = permutation_importance(
                champ_tr, Xs, y_te.loc[Xs.index], scoring=scoring,
                n_repeats=5 if X.shape[1] <= 40 else 3, random_state=42, n_jobs=1)
            importance = pd.DataFrame({
                "feature": X.columns, "importance": pi.importances_mean, "std": pi.importances_std,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            importance = None

    if progress:
        progress(1.0, "Done")

    out = {
        "task": task, "report": report, "leaderboard": leaderboard, "baseline": baseline,
        "champion_name": champion_name, "champion_test_metrics": _metrics(task, y_te, y_pred),
        "pipeline": final_pipe, "y_test": y_te.to_numpy(), "y_pred": y_pred,
        "clean_df": clean, "target": target, "feature_columns": X.columns.tolist(),
        "n_train": len(X_tr), "n_test": len(X_te),
        "skipped_models": skipped_models,
        "gap": gap, "time_ordered": bool(time_ordered), "select_percent": int(select_percent),
        "tuning": tuning, "importance": importance, "meta": {"source": "upload"},
        "warnings": run_warnings,
    }
    if task == "classification":
        labels = sorted(pd.unique(y))
        out["labels"] = [str(l) for l in labels]
        out["confusion"] = confusion_matrix(y_te, y_pred, labels=labels)
    return out


# ------------------------------------------------------------------
# 7. Export / import
# ------------------------------------------------------------------
def make_bundle(result) -> dict:
    """Everything needed to reuse the model later."""
    return {
        "pipeline": result["pipeline"],
        "task": result["task"],
        "target": result["target"],
        "feature_columns": result["feature_columns"],
        "champion": result["champion_name"],
        "meta": result.get("meta", {}),
    }


def export_pipeline_bytes(result) -> bytes:
    """Serialise the full pipeline (preprocessing + model) with metadata to .pkl bytes."""
    buf = io.BytesIO()
    joblib.dump(make_bundle(result), buf)
    return buf.getvalue()


def load_bundle(raw: bytes) -> dict:
    """Load a .pkl made by export_pipeline_bytes. Only load files you trust (pickle can run code)."""
    bundle = joblib.load(io.BytesIO(raw))
    need = {"pipeline", "task", "target", "feature_columns"}
    if not isinstance(bundle, dict) or not need <= set(bundle):
        raise ValueError("This file is not an AutoML Studio model bundle.")
    bundle.setdefault("meta", {})
    bundle.setdefault("champion", "unknown")
    return bundle


def model_card_markdown(result) -> str:
    """Short human-readable summary of a trained model."""
    rep = result["report"]
    m = result["champion_test_metrics"]
    tun = result.get("tuning")
    lines = [
        f"# Model card: {result['champion_name']}",
        "",
        f"- Task: {result['task']}",
        f"- Target: `{result['target']}`",
        f"- Rows used: {rep['rows_out']:,} (train {result['n_train']:,} / test {result['n_test']:,})",
        f"- Split: {'chronological' if result.get('time_ordered') else 'random'}"
        + (f", purge gap {result['gap']} rows" if result.get("gap") else ""),
        f"- Features ({len(result['feature_columns'])}): " + ", ".join(f"`{c}`" for c in result["feature_columns"]),
        f"- Date columns expanded: {', '.join(rep.get('datetime_columns', [])) or 'none'}",
        f"- Feature selection: top {result.get('select_percent', 100)}%",
        "",
        "## Test metrics (unseen data)",
        *[f"- {k}: {v:.4f} (baseline {result['baseline'][k]:.4f})" for k, v in m.items()],
    ]
    if tun:
        lines += ["", "## Hyperparameter tuning",
        f"- {tun['metric']} before: {tun['cv_before']:.4f}, after: {tun['cv_after']:.4f}"
        f" ({'adopted' if tun['adopted'] else 'not adopted'})",
        f"- Best params: {tun['best_params']}"]
    meta = result.get("meta", {})
    if meta.get("source") == "binance":
        lines += ["", "## Market data",
        f"- {meta.get('symbol')} {meta.get('interval')} candles, predicting {meta.get('horizon')} candles ahead"]
    lines += ["", "_Not financial advice. Short-term price prediction is close to random._"]
    return "\n".join(lines)


# ------------------------------------------------------------------
# 8. Market evaluation (for future-return targets)
# ------------------------------------------------------------------
def directional_accuracy(y_true, y_pred) -> float:
    """Share of rows where the predicted sign equals the actual sign (zero returns ignored)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = yt != 0
    return float(np.mean(np.sign(yt[m]) == np.sign(yp[m]))) if m.any() else float("nan")


def market_backtest(y_true, y_pred, horizon, fee=0.001) -> dict:
    """Long/flat strategy: go long for `horizon` candles when the predicted return is > 0.

    Uses non-overlapping periods (every `horizon`-th test row) so returns can be chained.
    `fee` is charged per side, i.e. 2*fee for every period in which a position is held.
    Buy-and-hold is the same non-overlapping chain without trading.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    idx = np.arange(0, len(yt), max(int(horizon), 1))
    r, p = yt[idx], yp[idx]
    pos = p > 0
    trade_ret = np.where(pos, r - 2 * fee, 0.0)
    strat = np.concatenate([[1.0], np.cumprod(1 + trade_ret)])
    hold = np.concatenate([[1.0], np.cumprod(1 + r)])
    peak = np.maximum.accumulate(strat)
    n_trades = int(pos.sum())
    return {
        "equity": pd.DataFrame({"period": np.arange(len(strat)), "Strategy": strat, "Buy & hold": hold}),
        "n_periods": int(len(r)), "n_trades": n_trades,
        "strategy_return": float(strat[-1] - 1), "buy_hold_return": float(hold[-1] - 1),
        "win_rate": float(np.mean(r[pos] > 2 * fee)) if n_trades else float("nan"),
        "max_drawdown": float(np.min(strat / peak - 1)),
    }


# ------------------------------------------------------------------
# 9. Live market helpers (XRP/USDT from Binance)
# ------------------------------------------------------------------
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]

MARKET_FEATURES = ["Return_1", "Return_3", "Return_6", "Return_12", "MA_5_Ratio", "MA_20_Ratio",
                   "Volatility_10", "RSI_7", "RSI_14", "Candle_Range", "Body_Ratio", "Volume_Change"]
_WARMUP_COLS = ["Return_12", "MA_20_Ratio", "Volatility_10", "RSI_14", "Volume_Change"]


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


def build_market_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Technical features from OHLCV. Every feature at row t uses only candles up to t (no look-ahead)."""
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
    return df.replace([np.inf, -np.inf], np.nan)


def market_to_dataset(candles: pd.DataFrame, horizon=12, task="regression", threshold=0.008):
    """Engineer features from OHLCV and add a future-return target (no leakage in features)."""
    df = build_market_features(candles)
    c = df["close"]
    fut = c.shift(-horizon) / c - 1
    if task == "regression":
        df["target_future_return"] = fut
        target = "target_future_return"
    else:
        df["target_direction"] = np.where(fut > threshold, "UP", np.where(fut < -threshold, "DOWN", "NEUTRAL"))
        df.loc[fut.isna(), "target_direction"] = np.nan
        target = "target_direction"
    df = df.dropna(subset=[target] + _WARMUP_COLS)
    df = df.drop(columns=["open", "high", "low", "close", "volume", "time"])  # raw price levels are non-stationary
    return df.reset_index(drop=True), target


def latest_market_features(candles: pd.DataFrame, n=5) -> pd.DataFrame:
    """Feature rows for the most recent closed candles (includes `time` and `close` for display)."""
    feats = build_market_features(candles).dropna(subset=_WARMUP_COLS)
    return feats.tail(n).reset_index(drop=True)
