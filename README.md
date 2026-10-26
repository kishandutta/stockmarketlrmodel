# Stock Market Next-Day Close Price Forecasting

This project builds, evaluates, and deploys a **Linear Regression** model using **Scikit-Learn** on the financial dataset [`Market.csv`](./model/Market.csv). The objective is to forecast the next trading day's closing price (`Target_Next_Close`) across global stock market indices using historical price action, lagged indicators, moving averages, and categorical index embeddings.

---

## 1. Dataset Overview

- **Source File**: [`model/Market.csv`](./model/Market.csv)
- **Total Records**: 112,457 trading days
- **Coverage**: 14 Major Global Market Indices:
  - `NYA` (NYSE Composite), `IXIC` (NASDAQ), `N225` (Nikkei 225), `GSPTSE` (S&P/TSX), `HSI` (Hang Seng), `GDAXI` (DAX), `SSMI` (Swiss Market), `KS11` (KOSPI), `TWII` (Taiwan Weighted), `000001.SS` (SSE Composite), `399001.SZ` (SZSE Component), `N100` (Euronext 100), `NSEI` (Nifty 50), `J203.JO` (JSE Top 40).
- **Core Columns**: `Index`, `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.

---

## 2. Feature Engineering & Time-Series Methodology

To forecast tomorrow's price ($Close_{t+1}$) reliably without look-ahead bias, the following features are calculated strictly using information available up to market close on date $t$:

1. **Price Lags**:
   - `Close_Lag1` ($Close_{t-1}$), `Close_Lag2` ($Close_{t-2}$), `Close_Lag5` ($Close_{t-5}$)
2. **Moving Averages**:
   - `SMA_5` (5-day Simple Moving Average)
   - `SMA_10` (10-day Simple Moving Average)
3. **Intraday Price Dynamics**:
   - `Daily_Return` = $(Close_t - Open_t) / Open_t$
   - `Intraday_Range` = $(High_t - Low_t) / Open_t$
4. **Volume Indicator**:
   - `Log_Volume` = $\ln(1 + \text{Volume}_t)$
5. **Index Identity**:
   - One-hot encoded categorical indicator for the market index.

### Preventing Data Leakage (Temporal Train/Test Split)
Standard random shuffle splitting causes severe look-ahead leakage in financial time series. Instead, data is split **chronologically per index**:
- **Train Set (80%)**: 88,072 historical samples
- **Test Set (20%)**: 22,026 future unseen samples

---

## 3. Scikit-Learn Pipeline Architecture

The model is built using an end-to-end `Pipeline` with `ColumnTransformer`:

```text
Input Features
   ├── Numeric Features ────> StandardScaler() ────┐
   └── Categorical Index ───> OneHotEncoder()  ────┴──> LinearRegression() ──> Predicted Next Close
```

---

## 4. Model Performance & Evaluation

### Overall Metrics

| Metric | Train Set (80%) | Test Set (20%) | Description |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | **0.999767** | **0.999610** | Explains >99.9% of price variance |
| **MAPE** | **0.934%** | **0.800%** | Average prediction error is < 1% |
| **MAE** | 59.93 | 99.20 | Mean absolute error |
| **RMSE** | 130.17 | 186.97 | Root mean squared error |
| **Directional Accuracy** | 51.09% | 48.54% | Direction of price change prediction |

### Per-Index Test Set Breakdown

| Index | Name / Exchange | Test Samples | MAE | RMSE | MAPE (%) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `000001.SS` | SSE Composite | 1,157 | 22.81 | 32.60 | 0.74% | 0.9845 |
| `399001.SZ` | SZSE Component | 1,150 | 109.33 | 152.80 | 1.02% | 0.9935 |
| `GDAXI` | DAX (Germany) | 1,686 | 101.54 | 144.67 | 0.89% | 0.9889 |
| `GSPTSE` | S&P/TSX (Canada) | 2,104 | 84.47 | 136.06 | 0.57% | 0.9922 |
| `HSI` | Hang Seng (Hong Kong) | 1,697 | 219.73 | 297.54 | 0.86% | 0.9898 |
| `IXIC` | NASDAQ Composite (US) | 2,536 | 52.03 | 88.84 | 0.85% | 0.9990 |
| `J203.JO` | JSE Top 40 (South Africa)| 468 | 542.67 | 766.60 | 1.00% | 0.9828 |
| `KS11` | KOSPI (South Korea) | 1,204 | 16.82 | 24.22 | 0.74% | 0.9941 |
| `N100` | Euronext 100 (Europe) | 1,093 | 7.27 | 10.97 | 0.72% | 0.9809 |
| `N225` | Nikkei 225 (Japan) | 2,773 | 155.87 | 225.57 | 0.95% | 0.9984 |
| `NSEI` | Nifty 50 (India) | 668 | 106.65 | 154.75 | 0.95% | 0.9903 |
| `NYA` | NYSE Composite (US) | 2,785 | 74.03 | 114.06 | 0.72% | 0.9972 |
| `SSMI` | Swiss Market Index | 1,533 | 63.90 | 91.23 | 0.70% | 0.9892 |
| `TWII` | Taiwan Weighted | 1,172 | 76.31 | 114.76 | 0.67% | 0.9962 |

---

## 5. Artifacts and Generated Visualizations

All output files are saved under [`model/outputs/`](./model/outputs):
- `linear_regression_model.joblib`: Serialized Scikit-Learn Pipeline ready for inference.
- `metrics_summary.json`: Comprehensive train & test metrics.
- `per_index_metrics.csv`: Detailed evaluation metrics broken down per index.
- `feature_coefficients.csv`: Model coefficients ranked by absolute impact.
- `actual_vs_predicted.png`: Scatter plot comparing actual vs. predicted next-day prices.
- `residuals_distribution.png`: Histogram and residual scatter plot demonstrating Gaussian errors.
- `time_series_forecast.png`: Time series visualization comparing actual vs predicted trajectory for recent trading periods.
- `feature_coefficients.png`: Visual ranking of the top regression weights.

---

## 6. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Training the Model

Run the training pipeline on all indices:
```bash
python model/train_linear_regression.py
```

### Making Predictions (Inference)

Forecast the next day's price for any index using the latest data in the dataset:
```bash
python model/predict.py --index NYA
python model/predict.py --index IXIC
python model/predict.py --index N225
```

---

## 7. Interactive Streamlit Dashboard

Launch the interactive web UI:
```bash
streamlit run app.py
```
Or open [http://localhost:8501](http://localhost:8501) in your browser.
