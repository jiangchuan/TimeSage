import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generate synthetic time series
n = 400
t = np.arange(n)
y = 50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1, n)
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Create sequences
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window = 30
X, y_seq = create_sequences(y_scaled, window)
X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_torch = torch.tensor(y_seq, dtype=torch.float32)

# Split
split = int(0.8 * len(X))
X_train, X_test = X_torch[:split], X_torch[split:]
y_train, y_test = y_torch[:split], y_torch[split:]

# Define LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# Define TCN
class TCN(nn.Module):
    def __init__(self, input_size=1, channels=[32], kernel_size=3):
        super().__init__()
        layers = []
        for i, c in enumerate(channels):
            dilation = 2 ** i
            layers.append(nn.Conv1d(input_size, c, kernel_size, padding=dilation*(kernel_size-1), dilation=dilation))
            layers.append(nn.ReLU())
            input_size = c
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(channels[-1], 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)[:, :, -1]
        return self.linear(out)

# Training loop
def train(model):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(50):
        opt.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        opt.step()
    model.eval()
    return model(X_test).detach().numpy()

# Train models
lstm = LSTMModel()
lstm_pred = train(lstm)
tcn = TCN()
tcn_pred = train(tcn)

# Rescale predictions
lstm_pred_rescaled = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
tcn_pred_rescaled = scaler.inverse_transform(tcn_pred.reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(y_true, label="True")
plt.plot(lstm_pred_rescaled, label="LSTM", linestyle="--")
plt.plot(tcn_pred_rescaled, label="TCN", linestyle="--")
plt.title("LSTM vs TCN Forecast")
plt.legend()
plt.show()

# RMSE
print("LSTM RMSE:", np.sqrt(mean_squared_error(y_true, lstm_pred_rescaled)))
print("TCN RMSE:", np.sqrt(mean_squared_error(y_true, tcn_pred_rescaled)))
