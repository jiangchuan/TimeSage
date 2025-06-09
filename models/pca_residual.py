import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate structured data
np.random.seed(42)
n, p = 200, 10
X = np.random.randn(n, p)
X[:, 0] = X[:, 1] + X[:, 2] + np.random.normal(0, 0.1, size=n)  # add collinearity
beta = np.random.randn(p)
y = X @ beta + np.random.normal(0, 1, n)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit OLS
model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
residuals = y_test - y_pred

# PCA on test set
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)

# Scatter residuals over PC space
plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=residuals, cmap='coolwarm', edgecolor='k')
plt.colorbar(sc, label='Residual')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('Residuals Projected onto Top 2 PCs')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.tight_layout()
plt.show()
