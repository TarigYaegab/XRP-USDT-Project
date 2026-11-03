# 🤖 AutoML Studio — XRP/USDT

An end-to-end AutoML app built with Streamlit. A non-technical user uploads a CSV (or pulls live XRP/USDT candles from Binance), picks a target column, and gets a fully trained, compared and ready-to-use **champion model** as a single downloadable `.pkl` file, with no code.

> ⚠️ **Disclaimer:** this is a learning/engineering project, not financial advice. Short-term price-return prediction is close to random; the app warns you when a model is no better than predicting the average.

---

## ✨ Features

- **Two data sources:** upload a `.csv`, or fetch live XRP/USDT history from Binance.
- **Live Market page:** candlestick chart, last price, change and volume, with a selectable interval.
- **Automatic task detection:** classification or regression, with a manual override.
- **Automatic cleaning:** duplicates, rows with no target, constant / ID-like / mostly-empty columns, numeric-looking text converted to numbers.
- **Automatic preprocessing:** missing-value imputation, outlier clipping, scaling, one-hot encoding and feature selection.
- **Multiple models compared** with cross-validation, then a **Champion** is picked.
- **Honest evaluation:** a baseline (predict the mean / most common class) is shown next to the test metrics.
- **Visuals:** correlation matrix, plus a confusion matrix (classification) or actual-vs-predicted scatter plot (regression).
- **Export:** the whole pipeline (preprocessing + model) as one `.pkl`.

---

## 🚀 Quick start

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
streamlit run automl_studio.py
```

Then open http://localhost:8501.

To open it from another device on the same network:

```bash
streamlit run automl_studio.py --server.address 0.0.0.0
```

---

## 🧭 How to use

### Live Market page
Pick an interval (1m–1d) and number of candles to see the XRP/USDT market. Data is cached for 60 seconds; use **Refresh** to reload.

### Model Dashboard page
1. **Data**
   - *Upload CSV*, or
   - *Live market (XRP/USDT)*: choose interval, history (30/90/180 days) and how many candles ahead to predict. Pick **regression** (future return) or **classification** (UP / DOWN / NEUTRAL with a threshold), then click *Fetch & prepare live data*.
2. **Target & settings**
   - Choose the target column; the detected task is shown.
   - *Advanced:* task override, time-series split, test size, CV folds, and **optional extra models** (XGBoost, LightGBM).
   - Feature-selection percentage and outlier fence (× IQR) sliders.
3. **Run AutoML**
4. **Results:** cleaning report, champion and metrics vs. baseline, leaderboard, correlation matrix, confusion matrix or scatter plot.
5. **Export:** download `champion_pipeline.pkl`.

---

## ⚙️ How it works

| Stage | What happens |
|---|---|
| **Task detection** | Non-numeric / boolean target → classification. Integer target with ≤10 unique values (or <2% unique) → classification. Otherwise regression. |
| **Cleaning** | `inf` → NaN; drop duplicates; drop rows with missing target; drop columns that are >60% missing, constant, or ID-like text; convert numeric-looking text to numbers. |
| **Preprocessing** (inside one sklearn `Pipeline`, learned on training data only) | Numeric: median impute → IQR outlier clipping (`Winsorizer`) → `StandardScaler`. Categorical: most-frequent impute → `OneHotEncoder` (rare categories grouped). Then `VarianceThreshold` → `SelectPercentile` (`f_classif` / `f_regression`). |
| **Split** | Random (stratified for classification), or chronological when *time series* is ticked. |
| **Model selection** | Models are ranked by cross-validation on the **training set only** (F1-macro for classification, RMSE for regression; `TimeSeriesSplit` for time series). The best is the **Champion**; test metrics are computed on unseen data. |
| **Export** | The champion is refit on all data and saved with `joblib`. |

### Models

| | Classification | Regression |
|---|---|---|
| Linear | LogisticRegression | Ridge |
| Trees | RandomForest | RandomForest |
| Boosting | HistGradientBoosting | HistGradientBoosting |
| Optional | XGBoost, LightGBM | XGBoost, LightGBM |

XGBoost and LightGBM are optional: if they are not installed the app still works and skips them with a warning. (XGBoost does not support class weighting inside this pipeline, so it does not compensate for imbalanced classes; the other classifiers use balanced weights.)

### Metrics
- **Classification:** Accuracy, Balanced Accuracy, F1 (macro).
- **Regression:** MAE, RMSE, R².

---

## 📈 Live-market features

`market_to_dataset` builds these features from Binance candles:

- Returns over 1, 3, 6 and 12 candles
- Price ratio to the 5- and 20-candle moving averages
- 10-candle volatility
- RSI (7 and 14)
- Candle range and body ratio
- Volume change

**Target:** `close[t+N] / close[t] − 1` (regression), or UP / DOWN / NEUTRAL by a threshold (classification). Raw open/high/low/close/volume levels are dropped because they are non-stationary and would let the model "cheat". Live data always uses a chronological split.

Candles come from the public Binance endpoint `https://data-api.binance.vision/api/v3/klines` (no API key needed), paginated, with the still-open candle removed.

---

## 📦 Using the exported model

```python
import joblib
import pandas as pd

bundle = joblib.load("champion_pipeline.pkl")   # keep automl_engine.py importable
new_df = pd.read_csv("new_data.csv")

pred = bundle["pipeline"].predict(new_df[bundle["feature_columns"]])
```

The bundle contains: `pipeline`, `task`, `target`, `feature_columns`, `champion`.

Notes:
- `automl_engine.py` must be importable when loading, because the pipeline uses the custom `Winsorizer`.
- If the champion is XGBoost or LightGBM, those packages must be installed where you load it.
- Use the same scikit-learn version that trained the model.

---

## 🗂 Project structure

```
automl_engine.py     # ML logic: detection, cleaning, preprocessing, models, export, Binance helpers
automl_studio.py     # Streamlit UI (Live Market + Model Dashboard)
requirements.txt     # dependencies
README.md
```

---

## ⚠️ Known limitations

- No hyperparameter tuning (fixed settings per model).
- The UI has a "Predict with a trained champion" heading but no prediction form yet; use the exported `.pkl` as shown above.
- Outlier clipping uses a fixed IQR rule, with no per-column control.
- Datasets with fewer than ~50 rows, or classes with fewer than 2 rows, will raise an error.
- `mlflow` is listed in `requirements.txt` but is not used by the code.

### Common pitfalls
- Don't choose a timestamp column as the target (it predicts nothing useful).
- Don't use same-candle open/high/low/close columns to predict `close`: correlation is ≈1.0 and the score is meaningless. Use the *Live market* option, which builds a proper future target.
- Use 1h candles over 90–180 days; one day of 5m candles (~288 rows) is too little to train on.

---

## 🛣 Roadmap

- [ ] Prediction tab: load a `.pkl` and predict on new data or the latest candle
- [ ] Hyperparameter search for the champion
- [ ] Optional MLflow run tracking
- [ ] Unit tests for `detect_task`, `clean_data` and `run_automl`

---

## 🧰 Tech stack

Python · Streamlit · scikit-learn · XGBoost · LightGBM · pandas · NumPy · Plotly · Binance public market data
