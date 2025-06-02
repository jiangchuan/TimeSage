import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic time series: trend + seasonality + noise
np.random.seed(42)
n = 365
t = np.arange(n)
trend = 0.05 * t
seasonality = 10 * np.sin(2 * np.pi * t / 30)  # ~monthly
noise = np.random.normal(0, 1, n)
y = 50 + trend + seasonality + noise

df = pd.DataFrame({'y': y}, index=pd.date_range(start='2022-01-01', periods=n))

# Create lag features
for lag in range(1, 8):  # lag1 to lag7
    df[f'lag_{lag}'] = df['y'].shift(lag)

# Add rolling mean feature
df['rolling_mean_7'] = df['y'].rolling(window=7).mean()

# Add day-of-week as categorical feature
df['day_of_week'] = df.index.dayofweek

# Drop rows with NaNs due to lag/rolling
df.dropna(inplace=True)

# Prepare features and target
features = [col for col in df.columns if col != 'y']
X = df[features]
y = df['y']

# Train/test split (last 30 days for test)
X_train, X_test = X[:-30], X[-30:]
y_train, y_test = y[:-30], y[-30:]

# Train XGBoost model
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(df.index[-60:], y[-60:], label="Actual", linewidth=2)
plt.plot(df.index[-30:], y_pred, label="XGBoost Forecast", linestyle='--', linewidth=2)
plt.axvline(df.index[-30], color='gray', linestyle=':', label='Train/Test Split')
plt.title("XGBoost Forecasting with Lag + Rolling Features")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'rmse = {rmse:.2f}')
