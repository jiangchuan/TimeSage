import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate synthetic ARIMA(2,1,1) data
np.random.seed(42)
n = 200
e = np.random.normal(0, 1, n)
y = np.zeros(n)

# Generate ARIMA(2,1,1): y_t = 2y_{t-1} - 0.5y_{t-2} + e_t + 0.7e_{t-1}
for t in range(2, n):
    y[t] = 2 * y[t - 1] - 0.5 * y[t - 2] + e[t] + 0.7 * e[t - 1]

# Integrated: take first difference to get stationary series
y_diff = pd.Series(y).diff().dropna()

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y_diff, lags=30, ax=axes[0])
axes[0].set_title("ACF of Differenced Series")
plot_pacf(y_diff, lags=30, ax=axes[1])
axes[1].set_title("PACF of Differenced Series")

plt.tight_layout()
plt.show()
