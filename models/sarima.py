# Re-run the code after execution environment reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Generate seasonal ARIMA(1,1,1)(1,1,1,12) series
np.random.seed(42)
n = 200
seasonal_period = 12
e = np.random.normal(0, 1, n)
y = np.zeros(n)

# Simulate SARIMA(1,1,1)(1,1,1,12)
for t in range(seasonal_period + 2, n):
    y[t] = (
            y[t - 1]
            - y[t - 2]
            + y[t - seasonal_period]
            - y[t - seasonal_period - 1]
            + e[t]
            + 0.5 * e[t - 1]
            + 0.4 * e[t - seasonal_period]
    )

series = pd.Series(y, index=pd.date_range("2000-01", periods=n, freq="M"))

# Decompose to show trend/seasonality
decomposition = seasonal_decompose(series, model='additive', period=seasonal_period)

# ADF test on original and differenced data
adf_raw = adfuller(series.dropna())
differenced = series.diff().dropna()
adf_diff1 = adfuller(differenced.dropna())
seasonally_differenced = differenced.diff(seasonal_period).dropna()
adf_diff1_seasonal = adfuller(seasonally_differenced)

# Plot time series and seasonal decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(series)
axes[0].set_title("Original Seasonal Time Series")
decomposition.trend.plot(ax=axes[1], title="Trend")
decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
decomposition.resid.plot(ax=axes[3], title="Residual")

plt.tight_layout()
plt.show()

# Plot ACF and PACF after differencing
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(seasonally_differenced, lags=40, ax=axes[0])
axes[0].set_title("ACF after Seasonal Differencing")
plot_pacf(seasonally_differenced, lags=40, ax=axes[1])
axes[1].set_title("PACF after Seasonal Differencing")

plt.tight_layout()
plt.show()

# Return ADF p-values for inspection
(adf_raw[1], adf_diff1[1], adf_diff1_seasonal[1])
