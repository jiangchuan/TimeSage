import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Generate synthetic time series: trend + seasonality + noise
np.random.seed(42)
n = 365  # daily data for 1 year
t = np.arange(n)
trend = 0.05 * t
seasonality = 10 * np.sin(2 * np.pi * t / 30)  # monthly seasonality
noise = np.random.normal(0, 1, n)
y = 50 + trend + seasonality + noise

# Create DataFrame
df = pd.DataFrame({'ds': pd.date_range(start='2022-01-01', periods=n), 'y': y})

# Apply FFT
y_fft = fft(df['y'].values)
frequencies = np.fft.fftfreq(n)

# Filter: Keep top-k largest amplitudes
k = 10
amplitudes = np.abs(y_fft)
indices = np.argsort(amplitudes)[-k:]  # get top k frequencies
filtered_fft = np.zeros_like(y_fft)
filtered_fft[indices] = y_fft[indices]

# Inverse FFT for smoothed reconstruction
y_smooth = np.real(ifft(filtered_fft))

# Extrapolate future values using sinusoidal components
future_steps = 60
t_future = np.arange(n + future_steps)
reconstructed_future = np.zeros_like(t_future, dtype=float)
for idx in indices:
    freq = frequencies[idx]
    amplitude = np.abs(y_fft[idx]) / n
    phase = np.angle(y_fft[idx])
    reconstructed_future += amplitude * n * np.cos(2 * np.pi * freq * t_future + phase)

# Plot original, smoothed, and forecast
plt.figure(figsize=(14, 6))
plt.plot(df['ds'], df['y'], label="Original")
plt.plot(df['ds'], y_smooth, label=f"Top-{k} Frequency Reconstruction", linewidth=2)
future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
plt.plot(np.concatenate([df['ds'], future_dates]), reconstructed_future, label="FFT Forecast", linestyle="--")
plt.axvline(df['ds'].iloc[-1], color='gray', linestyle=':', label="Forecast Start")
plt.title("FFT-Based Time Series Decomposition and Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
