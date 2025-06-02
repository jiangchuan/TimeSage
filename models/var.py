import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Create synthetic data: two interrelated time series
np.random.seed(42)
n = 200
e1 = np.random.normal(0, 1, n)
e2 = np.random.normal(0, 1, n)

y1 = np.zeros(n)
y2 = np.zeros(n)

# y1 influenced by its own lag and lag of y2
# y2 influenced by its own lag and lag of y1
for t in range(1, n):
    y1[t] = 0.6 * y1[t - 1] + 0.3 * y2[t - 1] + e1[t]
    y2[t] = 0.2 * y1[t - 1] + 0.5 * y2[t - 1] + e2[t]

df = pd.DataFrame({'y1': y1, 'y2': y2}, index=pd.date_range("2000-01-01", periods=n, freq="M"))

# Fit VAR model
model = VAR(df)
results = model.fit(maxlags=4, ic='aic')  # automatic lag selection

# Forecast the next 10 periods
forecast = results.forecast(df.values[-results.k_ar:], steps=10)
forecast_df = pd.DataFrame(forecast, columns=['y1', 'y2'])

# Impulse Response Function (IRF)
irf = results.irf(10)

# Plot the original series
df.plot(figsize=(12, 5), title="Synthetic Time Series (y1 and y2)")
plt.grid(True)
plt.show()

# Plot Impulse Response
irf.plot(orth=False)
plt.tight_layout()
plt.show()
