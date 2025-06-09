# Re-import after code execution environment reset
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Simulate sparse early data (cold start)
np.random.seed(42)
prices = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
true_elasticity = -1.5
baseline_demand = 1000000

# Simulated demand: demand = baseline * (price / min_price) ^ elasticity + noise
demand = baseline_demand * (prices / prices[0]) ** true_elasticity
demand += np.random.normal(0, 20000, size=len(prices))
demand = np.clip(demand, 0, None)

# Bayesian model for log-log price elasticity
log_price = np.log(prices)
log_demand = np.log(demand)

with pm.Model() as elasticity_model:
    alpha = pm.Normal("alpha", mu=13.5, sigma=1.0)  # intercept
    beta = pm.Normal("beta", mu=-1.5, sigma=0.5)  # price elasticity prior
    sigma = pm.Exponential("sigma", 1.0)

    mu = alpha + beta * log_price
    observed = pm.Normal("log_demand", mu=mu, sigma=sigma, observed=log_demand)

    trace = pm.sample(10, tune=1000, target_accept=0.95, return_inferencedata=True)

# Posterior summary
az.plot_posterior(trace, var_names=["beta"], hdi_prob=0.9)
plt.title("Posterior Distribution of Price Elasticity (Beta)")
plt.show()
