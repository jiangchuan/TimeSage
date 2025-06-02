import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Create synthetic seasonal time series with trend
np.random.seed(42)
n = 120  # 10 years of monthly data
trend = np.linspace(10, 50, n)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 2, n)
y = trend + seasonality + noise
series = pd.Series(y, index=pd.date_range("2010-01", periods=n, freq="M"))

# Fit ETS model with additive trend and seasonality
model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

# Forecast next 24 months
forecast = fit.forecast(24)

# Plot original series and forecast
plt.figure(figsize=(14, 6))
plt.plot(series, label="Observed")
plt.plot(forecast, label="Forecast", linestyle="--")
plt.axvline(series.index[-1], color='gray', linestyle=':', label='Forecast Start')
plt.title("ETS(A,A,A) Forecast: Additive Trend + Additive Seasonality")
plt.legend()
plt.grid(True)
plt.show()
