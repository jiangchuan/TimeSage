import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from numpy.linalg import svd

# Generate synthetic time series
np.random.seed(42)
n = 365
t = np.arange(n)
y = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1, n)
ds = pd.date_range("2022-01-01", periods=n)
df = pd.DataFrame({"ds": ds, "y": y})

# Fit Prophet model with custom seasonality
model = Prophet(yearly_seasonality=False, weekly_seasonality=True)
model.add_seasonality(name='monthly', period=30, fourier_order=5)
model.fit(df)

# Extract seasonal design matrix
seasonal_features, *_ = model.make_all_seasonality_features(df[['ds']])
X = seasonal_features.values

# SVD decomposition
U, S, VT = svd(X, full_matrices=False)

# Plot spectrum
plt.figure(figsize=(10, 4))
plt.plot(S, marker='o')
plt.title("Singular Values of Prophet Seasonal Design Matrix")
plt.xlabel("Component")
plt.ylabel("Singular Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot first few left singular vectors (they correspond to time axis)
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(df['ds'].values, U[:, i], label=f"Left Singular Vector {i + 1}")
plt.title("Top 5 Left Singular Vectors (Temporal Patterns)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
