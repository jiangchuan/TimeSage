import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate data
np.random.seed(0)
n, p = 100, 30
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = np.random.randn(5)  # only first 5 features matter
y = X @ true_beta + np.random.normal(0, 1, n)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Standardize
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit models
ols = LinearRegression().fit(X_train_scaled, y_train)
ridge = Ridge(alpha=10).fit(X_train_scaled, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_scaled, y_train)

# Predict and compute residuals
models = {'OLS': ols, 'Ridge': ridge, 'Lasso': lasso}
plt.figure(figsize=(15, 4))

for i, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred
    plt.subplot(1, 3, i)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{name} Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')

plt.tight_layout()
plt.show()
