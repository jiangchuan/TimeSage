import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Generate synthetic data
n = 365
t = np.arange(n)
y = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1, n)
dates = pd.date_range("2022-01-01", periods=n)
df = pd.DataFrame({'ds': dates, 'y': y})

# Create lag features
df['y_lag1'] = df['y'].shift(1)
df['y_lag7'] = df['y'].shift(7)
df.dropna(inplace=True)

# Initialize model and add lagged regressors
model = Prophet()
model.add_regressor('y_lag1')
model.add_regressor('y_lag7')
model.fit(df[['ds', 'y', 'y_lag1', 'y_lag7']])

# Create future DataFrame
future = model.make_future_dataframe(periods=30)
full_df = pd.concat([df[['ds', 'y']], future[~future['ds'].isin(df['ds'])]], ignore_index=True)

# Fill future lags using recursive simulation from known values
for i in range(len(df), len(full_df)):
    if i - 1 < 0 or pd.isna(full_df.loc[i - 1, 'y']):
        full_df.at[i, 'y_lag1'] = np.nan
    else:
        full_df.at[i, 'y_lag1'] = full_df.loc[i - 1, 'y']

    if i - 7 < 0 or pd.isna(full_df.loc[i - 7, 'y']):
        full_df.at[i, 'y_lag7'] = np.nan
    else:
        full_df.at[i, 'y_lag7'] = full_df.loc[i - 7, 'y']

# Drop rows in the forecast frame that contain NaNs in any regressor
future_with_lags = full_df[['ds', 'y_lag1', 'y_lag7']].copy()
future_with_lags.dropna(inplace=True)

# Now safe to predict
forecast = model.predict(future_with_lags)

# Plot forecast
plt.figure(figsize=(14, 6))
plt.plot(df['ds'], df['y'], label='Observed')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
plt.axvline(df['ds'].iloc[-1], color='gray', linestyle=':', label='Forecast Start')
plt.title("Prophet Forecast with Lagged Regressors")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
