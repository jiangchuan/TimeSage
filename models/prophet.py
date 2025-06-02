import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: Create synthetic dataset
np.random.seed(42)
days = 3 * 365
date_range = pd.date_range(start="2020-01-01", periods=days, freq="D")

trend = 0.01 * np.arange(days)
seasonality = 5 * np.sin(2 * np.pi * np.arange(days) / 365)
noise = np.random.normal(0, 1, days)
y = 50 + trend + seasonality + noise

df = pd.DataFrame({'ds': date_range, 'y': y})

# Step 2: Define holidays
holidays = pd.DataFrame({
    'holiday': ['new_year', 'christmas'] * 3,
    'ds': pd.to_datetime(['2020-01-01', '2020-12-25',
                          '2021-01-01', '2021-12-25',
                          '2022-01-01', '2022-12-25']),
    'lower_window': 0,
    'upper_window': 1
})

# Step 3: Fit the model
model = Prophet(holidays=holidays, yearly_seasonality=True)
model.fit(df)

# Step 4: Forecast into the future
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# Step 5: Plot forecast
model.plot(forecast)
plt.title("Prophet Forecast with Trend, Seasonality, and Holidays")
plt.grid(True)
plt.show()

# Step 6: Plot components
model.plot_components(forecast)
plt.tight_layout()
plt.show()
