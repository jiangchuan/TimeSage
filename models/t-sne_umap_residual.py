import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate structured data
np.random.seed(42)
n, p = 200, 10
X = np.random.randn(n, p)
X[:, 0] = X[:, 1] + X[:, 2] + np.random.normal(0, 0.1, size=n)
beta = np.random.randn(p)
y = X @ beta + np.random.normal(0, 1, n)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model and compute residuals
model = LinearRegression().fit(scaler.transform(X_train), y_train)
residuals = y_test - model.predict(X_test_scaled)

# t-SNE projection
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X_test_scaled)

# UMAP projection
reducer = umap.UMAP(n_components=2, random_state=0)
X_umap = reducer.fit_transform(X_test_scaled)

# Plot function
def plot_embedding(embedding, title):
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=residuals, cmap='coolwarm', edgecolor='k', s=40)
    plt.colorbar(sc, label='Residual')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualizations
plot_embedding(X_tsne, "Residuals on t-SNE projection")
plot_embedding(X_umap, "Residuals on UMAP projection")
