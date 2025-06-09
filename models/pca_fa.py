from sklearn.decomposition import PCA, FactorAnalysis
import numpy as np

# Simulated data with latent structure
np.random.seed(0)
n, p, k = 500, 10, 3
F = np.random.randn(n, k)
loadings = np.random.randn(p, k)
noise = np.random.randn(n, p) * 0.5
X = F @ loadings.T + noise

# PCA
pca = PCA(n_components=k).fit(X)
print("PCA Explained Variance:", pca.explained_variance_ratio_)

# Factor Analysis
fa = FactorAnalysis(n_components=k).fit(X)
print("FA Estimated Loadings:\n", fa.components_.T)
