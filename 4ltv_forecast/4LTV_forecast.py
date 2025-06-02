# 💡 Forecast Lifetime Value (LTV)
# You’re asked to forecast customer LTV across user cohorts to support pricing decisions.
# You’re given signup date, usage patterns, and revenue per user over time.
# How would you forecast LTV? How do you handle uncertainty?

# Cohort-based survival models, churn curves (e.g., Kaplan-Meier, exponential decay)
# Customer-level modeling: RFM (Recency-Frequency-Monetary), time-aware regression
# Monte Carlo simulation or Bayesian LTV estimation
# Lifetime = ∑ expected revenue over time × survival probability
# Explain variance by cohort and product tier

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Survival analysis
from lifelines import KaplanMeierFitter, WeibullFitter, LogNormalFitter

# Machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


class CustomerLTVForecaster:
    """
    Comprehensive Customer Lifetime Value forecasting system with:
    1. Cohort-based analysis
    2. Survival modeling for churn prediction
    3. Revenue forecasting with uncertainty quantification
    4. Multi-model ensemble approach
    5. Scenario planning for pricing decisions
    """

    def __init__(self):
        self.survival_models = {}
        self.revenue_models = {}
        self.cohort_models = {}
        self.uncertainty_models = {}
        self.feature_importance = {}

    def generate_synthetic_customer_data(self, n_customers=50000, start_date='2022-01-01'):
        """Generate realistic customer data for LTV modeling"""

        np.random.seed(42)
        start_dt = pd.to_datetime(start_date)

        # Generate customer signup dates over 24 months
        signup_dates = pd.date_range(start_dt, start_dt + timedelta(days=730), freq='D')
        signup_weights = np.exp(np.linspace(0, 2, len(signup_dates)))  # Growing signups

        customers = []
        customer_id = 1

        for date in signup_dates:
            # Number of signups per day (varying by day of week, seasonality)
            base_signups = int(np.random.poisson(signup_weights[np.where(signup_dates == date)[0][0]] * 20))

            # Weekend effect (fewer signups)
            if date.weekday() >= 5:
                base_signups = int(base_signups * 0.7)

            # Holiday/seasonal effects
            month = date.month
            if month in [12, 1]:  # Holiday boost
                base_signups = int(base_signups * 1.4)
            elif month in [6, 7, 8]:  # Summer dip
                base_signups = int(base_signups * 0.8)

            for _ in range(base_signups):
                if customer_id > n_customers:
                    break

                # Customer characteristics
                customer = {
                    'customer_id': customer_id,
                    'signup_date': date,
                    'acquisition_channel': np.random.choice(['organic', 'paid_search', 'social', 'referral', 'email'],
                                                            p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                    'plan_type': np.random.choice(['free', 'basic', 'pro', 'enterprise'],
                                                  p=[0.4, 0.3, 0.25, 0.05]),
                    'company_size': np.random.choice(['1-10', '11-50', '51-200', '201-1000', '1000+'],
                                                     p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                    'industry': np.random.choice(['tech', 'finance', 'healthcare', 'retail', 'education', 'other'],
                                                 p=[0.25, 0.15, 0.12, 0.15, 0.13, 0.2]),
                    'geography': np.random.choice(['US', 'EU', 'APAC', 'Other'], p=[0.45, 0.25, 0.2, 0.1]),
                }

                customers.append(customer)
                customer_id += 1

            if customer_id > n_customers:
                break

        df_customers = pd.DataFrame(customers)

        # Generate usage and revenue data
        usage_data = []
        current_date = start_dt
        end_date = datetime.now().replace(day=1)  # Current month

        for _, customer in df_customers.iterrows():
            customer_start = customer['signup_date']

            # Determine customer lifecycle characteristics
            # Churn probability based on characteristics
            base_churn_rate = 0.05  # 5% monthly base churn

            churn_multipliers = {
                'free': 2.0, 'basic': 1.2, 'pro': 0.8, 'enterprise': 0.4,
                'organic': 0.8, 'paid_search': 1.0, 'social': 1.3, 'referral': 0.6, 'email': 1.1,
                '1-10': 1.4, '11-50': 1.0, '51-200': 0.8, '201-1000': 0.6, '1000+': 0.4
            }

            monthly_churn_rate = base_churn_rate * churn_multipliers.get(customer['plan_type'], 1.0) * \
                                 churn_multipliers.get(customer['acquisition_channel'], 1.0) * \
                                 churn_multipliers.get(customer['company_size'], 1.0)

            # Revenue characteristics
            revenue_base = {'free': 0, 'basic': 29, 'pro': 99, 'enterprise': 399}
            monthly_revenue_base = revenue_base[customer['plan_type']]

            # Company size affects revenue (enterprise customers pay more)
            size_multipliers = {'1-10': 1.0, '11-50': 1.2, '51-200': 1.5, '201-1000': 2.0, '1000+': 3.0}
            monthly_revenue_base *= size_multipliers[customer['company_size']]

            # Generate monthly data for this customer
            current_month = customer_start.replace(day=1)
            months_active = 0
            is_churned = False

            while current_month <= end_date and not is_churned:
                # Check for churn this month
                if np.random.random() < monthly_churn_rate:
                    is_churned = True
                    churn_date = current_month + timedelta(days=np.random.randint(1, 28))
                else:
                    churn_date = None

                # Usage patterns (affects revenue and churn)
                if months_active == 0:
                    usage_level = np.random.uniform(0.3, 1.0)  # Initial usage
                else:
                    # Usage evolution over time
                    prev_usage = usage_data[-1]['usage_level'] if usage_data else 0.5
                    usage_change = np.random.normal(0, 0.1)
                    usage_level = np.clip(prev_usage + usage_change, 0.1, 2.0)

                # Revenue calculation
                base_revenue = monthly_revenue_base * usage_level

                # Add noise and seasonality
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * current_month.month / 12)
                revenue_noise = np.random.normal(1, 0.15)
                actual_revenue = base_revenue * seasonal_factor * revenue_noise

                # Expansion revenue (upgrades)
                if months_active > 3 and np.random.random() < 0.05:  # 5% chance of upgrade
                    actual_revenue *= np.random.uniform(1.5, 3.0)

                usage_record = {
                    'customer_id': customer['customer_id'],
                    'month': current_month,
                    'months_since_signup': months_active,
                    'revenue': max(0, actual_revenue),
                    'usage_level': usage_level,
                    'is_churned': is_churned,
                    'churn_date': churn_date
                }

                usage_data.append(usage_record)

                if is_churned:
                    break

                current_month += timedelta(days=32)
                current_month = current_month.replace(day=1)
                months_active += 1

                # Reduce churn rate over time (survivors are stickier)
                monthly_churn_rate *= 0.98

        df_usage = pd.DataFrame(usage_data)

        # Merge customer characteristics with usage data
        df_final = df_usage.merge(df_customers, on='customer_id', how='left')

        # Add cohort information
        df_final['signup_cohort'] = df_final['signup_date'].dt.to_period('M')
        df_final['revenue_cohort'] = pd.cut(df_final.groupby('customer_id')['revenue'].transform('mean'),
                                            bins=[0, 10, 50, 150, 500, float('inf')],
                                            labels=['Low', 'Medium', 'High', 'Premium', 'Enterprise'])

        return df_final

    def prepare_cohort_analysis(self, df):
        """Prepare data for cohort-based LTV analysis"""

        # Create cohort table
        cohort_data = df.groupby(['signup_cohort', 'months_since_signup']).agg({
            'customer_id': 'nunique',
            'revenue': ['sum', 'mean'],
            'is_churned': 'sum'
        }).reset_index()

        # Flatten column names
        cohort_data.columns = ['signup_cohort', 'months_since_signup', 'active_customers',
                               'total_revenue', 'avg_revenue_per_customer', 'churned_customers']

        # Calculate cohort sizes
        cohort_sizes = df.groupby('signup_cohort')['customer_id'].nunique().reset_index()
        cohort_sizes.columns = ['signup_cohort', 'cohort_size']

        cohort_data = cohort_data.merge(cohort_sizes, on='signup_cohort')

        # Calculate retention rates
        cohort_data['retention_rate'] = cohort_data['active_customers'] / cohort_data['cohort_size']
        cohort_data['cumulative_revenue_per_customer'] = cohort_data.groupby('signup_cohort')[
            'avg_revenue_per_customer'].cumsum()

        return cohort_data

    def fit_survival_models(self, df):
        """Fit multiple survival models to predict customer churn"""

        # Prepare survival data
        survival_data = df.groupby('customer_id').agg({
            'months_since_signup': 'max',
            'is_churned': 'max',
            'signup_date': 'first',
            'acquisition_channel': 'first',
            'plan_type': 'first',
            'company_size': 'first',
            'industry': 'first',
            'revenue': 'mean'
        }).reset_index()

        survival_data['tenure'] = survival_data['months_since_signup'] + 1
        survival_data['event_observed'] = survival_data['is_churned']

        # Fit Kaplan-Meier for overall survival
        kmf = KaplanMeierFitter()
        kmf.fit(survival_data['tenure'], survival_data['event_observed'])
        self.survival_models['kaplan_meier'] = kmf

        # Fit parametric models
        models_to_fit = [
            ('weibull', WeibullFitter()),
            ('lognormal', LogNormalFitter())
        ]

        for name, model in models_to_fit:
            try:
                model.fit(survival_data['tenure'], survival_data['event_observed'])
                self.survival_models[name] = model
            except:
                print(f"Failed to fit {name} model")

        # Fit survival models by cohort characteristics
        cohort_survival = {}

        for channel in survival_data['acquisition_channel'].unique():
            channel_data = survival_data[survival_data['acquisition_channel'] == channel]
            if len(channel_data) > 50:  # Minimum sample size
                kmf_channel = KaplanMeierFitter()
                kmf_channel.fit(channel_data['tenure'], channel_data['event_observed'])
                cohort_survival[f'channel_{channel}'] = kmf_channel

        for plan in survival_data['plan_type'].unique():
            plan_data = survival_data[survival_data['plan_type'] == plan]
            if len(plan_data) > 50:
                kmf_plan = KaplanMeierFitter()
                kmf_plan.fit(plan_data['tenure'], plan_data['event_observed'])
                cohort_survival[f'plan_{plan}'] = kmf_plan

        self.survival_models['cohort_models'] = cohort_survival

        return survival_data

    def fit_revenue_models(self, df):
        """Fit models to predict monthly revenue per customer"""

        # Prepare features for revenue modeling
        revenue_features = df.copy()

        # Create lagged features
        revenue_features = revenue_features.sort_values(['customer_id', 'month'])
        revenue_features['prev_revenue'] = revenue_features.groupby('customer_id')['revenue'].shift(1)
        revenue_features['prev_usage'] = revenue_features.groupby('customer_id')['usage_level'].shift(1)

        # Calculate revenue trends
        revenue_features['revenue_growth'] = revenue_features.groupby('customer_id')['revenue'].pct_change()
        revenue_features['cumulative_revenue'] = revenue_features.groupby('customer_id')['revenue'].cumsum()

        # Encode categorical variables
        categorical_cols = ['acquisition_channel', 'plan_type', 'company_size', 'industry', 'geography']
        for col in categorical_cols:
            dummies = pd.get_dummies(revenue_features[col], prefix=col)
            revenue_features = pd.concat([revenue_features, dummies], axis=1)

        # Select features for modeling
        feature_cols = [col for col in revenue_features.columns if
                        col.startswith(tuple(categorical_cols)) or
                        col in ['months_since_signup', 'usage_level', 'prev_revenue', 'prev_usage']]

        feature_cols = list(
            set(feature_cols) - set(['acquisition_channel', 'plan_type', 'company_size', 'industry', 'geography']))

        # Remove rows with missing values
        model_data = revenue_features[feature_cols + ['revenue']].dropna()

        X = model_data[feature_cols]
        y = model_data['revenue']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

            self.revenue_models[name] = model

            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df

        self.feature_cols = feature_cols
        return model_data

    def forecast_ltv_probabilistic(self, df, forecast_months=6):
        """Generate probabilistic LTV forecasts with uncertainty quantification"""

        # Get unique customers
        customers = df.drop_duplicates('customer_id')
        ltv_forecasts = []

        for _, customer in customers.iterrows():
            customer_id = customer['customer_id']

            # Get customer history
            customer_history = df[df['customer_id'] == customer_id].sort_values('month')

            if len(customer_history) == 0:
                continue

            # Calculate current metrics
            current_tenure = customer_history['months_since_signup'].max()
            avg_monthly_revenue = customer_history['revenue'].mean()
            recent_revenue = customer_history['revenue'].iloc[-3:].mean() if len(
                customer_history) >= 3 else avg_monthly_revenue

            # Survival probability forecast
            survival_probs = []
            for month in range(1, forecast_months + 1):
                future_tenure = current_tenure + month

                # Use best available survival model
                if 'weibull' in self.survival_models:
                    survival_prob = self.survival_models['weibull'].survival_function_at_times(future_tenure).iloc[0]
                else:
                    survival_prob = self.survival_models['kaplan_meier'].survival_function_at_times(future_tenure).iloc[
                        0]

                survival_probs.append(survival_prob)

            # Revenue forecast with uncertainty
            revenue_scenarios = self._generate_revenue_scenarios(customer, customer_history, forecast_months)

            # Calculate LTV scenarios
            ltv_scenarios = {}
            for scenario_name, revenues in revenue_scenarios.items():
                ltv = 0
                for month, (revenue, survival_prob) in enumerate(zip(revenues, survival_probs)):
                    # Discount future revenue
                    discount_factor = (1 / (1 + 0.01)) ** month  # 1% monthly discount rate
                    ltv += revenue * survival_prob * discount_factor

                ltv_scenarios[scenario_name] = ltv

            # Calculate confidence intervals
            ltv_values = list(ltv_scenarios.values())
            ltv_forecast = {
                'customer_id': customer_id,
                'current_tenure': current_tenure,
                'avg_monthly_revenue': avg_monthly_revenue,
                'ltv_p10': np.percentile(ltv_values, 10),
                'ltv_p25': np.percentile(ltv_values, 25),
                'ltv_median': np.percentile(ltv_values, 50),
                'ltv_p75': np.percentile(ltv_values, 75),
                'ltv_p90': np.percentile(ltv_values, 90),
                'ltv_mean': np.mean(ltv_values),
                'ltv_std': np.std(ltv_values),
                'survival_prob_12m': survival_probs[11] if len(survival_probs) >= 12 else None,
                'survival_prob_24m': survival_probs[23] if len(survival_probs) >= 24 else None,
                **{f'ltv_{k}': v for k, v in ltv_scenarios.items()}
            }

            # Add customer characteristics
            for col in ['acquisition_channel', 'plan_type', 'company_size', 'industry', 'signup_cohort']:
                ltv_forecast[col] = customer[col]

            ltv_forecasts.append(ltv_forecast)

        return pd.DataFrame(ltv_forecasts)

    def _generate_revenue_scenarios(self, customer, history, forecast_months, n_scenarios=100):
        """Generate multiple revenue scenarios for uncertainty quantification"""

        recent_revenue = history['revenue'].iloc[-3:].mean() if len(history) >= 3 else history['revenue'].mean()
        revenue_volatility = history['revenue'].std() / history['revenue'].mean() if history[
                                                                                         'revenue'].mean() > 0 else 0.3

        scenarios = {}

        # Conservative scenario (10th percentile)
        conservative_growth = -0.02  # 2% monthly decline
        conservative_revenues = []
        current_revenue = recent_revenue * 0.8
        for month in range(forecast_months):
            current_revenue *= (1 + conservative_growth + np.random.normal(0, revenue_volatility * 0.5))
            current_revenue = max(0, current_revenue)
            conservative_revenues.append(current_revenue)
        scenarios['conservative'] = conservative_revenues

        # Base case scenario (median)
        base_growth = 0.01  # 1% monthly growth
        base_revenues = []
        current_revenue = recent_revenue
        for month in range(forecast_months):
            current_revenue *= (1 + base_growth + np.random.normal(0, revenue_volatility))
            current_revenue = max(0, current_revenue)
            base_revenues.append(current_revenue)
        scenarios['base'] = base_revenues

        # Optimistic scenario (90th percentile)
        optimistic_growth = 0.05  # 5% monthly growth
        optimistic_revenues = []
        current_revenue = recent_revenue * 1.2
        for month in range(forecast_months):
            current_revenue *= (1 + optimistic_growth + np.random.normal(0, revenue_volatility * 1.5))
            current_revenue = max(0, current_revenue)
            optimistic_revenues.append(current_revenue)
        scenarios['optimistic'] = optimistic_revenues

        # Monte Carlo scenarios
        for i in range(n_scenarios):
            growth_rate = np.random.normal(0.01, 0.03)  # Random growth rate
            mc_revenues = []
            current_revenue = recent_revenue * np.random.uniform(0.8, 1.2)

            for month in range(forecast_months):
                # Add random shocks
                shock = np.random.choice([1.0, 1.5, 0.5], p=[0.9, 0.05, 0.05])  # 10% chance of shock
                current_revenue *= (1 + growth_rate + np.random.normal(0, revenue_volatility)) * shock
                current_revenue = max(0, current_revenue)
                mc_revenues.append(current_revenue)

            scenarios[f'mc_{i}'] = mc_revenues

        return scenarios

    def calculate_cohort_ltv(self, ltv_forecasts):
        """Calculate LTV by various cohorts for pricing analysis"""

        cohort_analysis = {}

        # By acquisition channel
        channel_ltv = ltv_forecasts.groupby('acquisition_channel').agg({
            'ltv_median': ['mean', 'std', 'count'],
            'ltv_p25': 'mean',
            'ltv_p75': 'mean',
            'survival_prob_12m': 'mean',
            'survival_prob_24m': 'mean'
        }).round(2)
        cohort_analysis['acquisition_channel'] = channel_ltv

        # By plan type
        plan_ltv = ltv_forecasts.groupby('plan_type').agg({
            'ltv_median': ['mean', 'std', 'count'],
            'ltv_p25': 'mean',
            'ltv_p75': 'mean',
            'survival_prob_12m': 'mean',
            'survival_prob_24m': 'mean'
        }).round(2)
        cohort_analysis['plan_type'] = plan_ltv

        # By company size
        size_ltv = ltv_forecasts.groupby('company_size').agg({
            'ltv_median': ['mean', 'std', 'count'],
            'ltv_p25': 'mean',
            'ltv_p75': 'mean',
            'survival_prob_12m': 'mean',
            'survival_prob_24m': 'mean'
        }).round(2)
        cohort_analysis['company_size'] = size_ltv

        # By signup cohort (time-based)
        signup_ltv = ltv_forecasts.groupby('signup_cohort').agg({
            'ltv_median': ['mean', 'std', 'count'],
            'ltv_p25': 'mean',
            'ltv_p75': 'mean',
            'survival_prob_12m': 'mean',
            'survival_prob_24m': 'mean'
        }).round(2)
        cohort_analysis['signup_cohort'] = signup_ltv

        return cohort_analysis

    def pricing_sensitivity_analysis(self, ltv_forecasts, current_prices):
        """Analyze pricing sensitivity and optimal pricing strategies"""

        pricing_analysis = {}

        for plan_type, current_price in current_prices.items():
            plan_customers = ltv_forecasts[ltv_forecasts['plan_type'] == plan_type]

            if len(plan_customers) == 0:
                continue

            # Calculate current LTV/CAC ratios
            current_ltv = plan_customers['ltv_median'].mean()

            # Price elasticity scenarios
            price_scenarios = np.arange(0.5, 2.1, 0.1)  # 50% to 200% of current price
            scenario_results = []

            for price_multiplier in price_scenarios:
                new_price = current_price * price_multiplier

                # Estimate demand impact (simplified price elasticity)
                if plan_type == 'free':
                    elasticity = -0.5  # Free plans less sensitive
                elif plan_type == 'enterprise':
                    elasticity = -0.3  # Enterprise less price sensitive
                else:
                    elasticity = -1.2  # Standard elasticity

                demand_change = elasticity * (price_multiplier - 1)
                retention_impact = min(0.1, abs(demand_change) * 0.5)  # Price affects retention

                # Adjust LTV for price and retention changes
                adjusted_ltv = current_ltv * price_multiplier * (1 + retention_impact)
                customer_volume_multiplier = 1 + demand_change

                total_value = adjusted_ltv * customer_volume_multiplier

                scenario_results.append({
                    'price_multiplier': price_multiplier,
                    'new_price': new_price,
                    'ltv_per_customer': adjusted_ltv,
                    'volume_multiplier': customer_volume_multiplier,
                    'total_value_index': total_value / current_ltv,
                    'demand_change_pct': demand_change * 100
                })

            pricing_analysis[plan_type] = pd.DataFrame(scenario_results)

        return pricing_analysis

    def plot_ltv_analysis(self, df, ltv_forecasts, cohort_analysis):
        """Create comprehensive LTV analysis visualizations"""

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # 1. LTV distribution
        ax1 = axes[0, 0]
        ax1.hist(ltv_forecasts['ltv_median'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(ltv_forecasts['ltv_median'].mean(), color='red', linestyle='--',
                    label=f'Mean: ${ltv_forecasts["ltv_median"].mean():.0f}')
        ax1.axvline(ltv_forecasts['ltv_median'].median(), color='orange', linestyle='--',
                    label=f'Median: ${ltv_forecasts["ltv_median"].median():.0f}')
        ax1.set_xlabel('LTV ($)')
        ax1.set_ylabel('Number of Customers')
        ax1.set_title('Customer LTV Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. LTV by acquisition channel
        ax2 = axes[0, 1]
        channel_data = ltv_forecasts.groupby('acquisition_channel')['ltv_median'].agg(['mean', 'std']).reset_index()
        bars = ax2.bar(channel_data['acquisition_channel'], channel_data['mean'],
                       yerr=channel_data['std'], capsize=5, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Acquisition Channel')
        ax2.set_ylabel('Average LTV ($)')
        ax2.set_title('LTV by Acquisition Channel')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, channel_data['mean']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f'${value:.0f}', ha='center', va='bottom')

        # 3. LTV by plan type
        ax3 = axes[0, 2]
        plan_data = ltv_forecasts.groupby('plan_type')['ltv_median'].agg(['mean', 'std']).reset_index()
        bars = ax3.bar(plan_data['plan_type'], plan_data['mean'],
                       yerr=plan_data['std'], capsize=5, color='lightcoral', alpha=0.7)
        ax3.set_xlabel('Plan Type')
        ax3.set_ylabel('Average LTV ($)')
        ax3.set_title('LTV by Plan Type')
        ax3.grid(True, alpha=0.3)

        for bar, value in zip(bars, plan_data['mean']):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f'${value:.0f}', ha='center', va='bottom')

        # 4. Survival curves
        ax4 = axes[1, 0]
        if 'kaplan_meier' in self.survival_models:
            kmf = self.survival_models['kaplan_meier']
            kmf.survival_function_.plot(ax=ax4, color='blue', linewidth=2)
        ax4.set_xlabel('Months')
        ax4.set_ylabel('Survival Probability')
        ax4.set_title('Customer Survival Curve')
        ax4.grid(True, alpha=0.3)

        # 5. LTV uncertainty (confidence intervals)
        ax5 = axes[1, 1]
        ltv_percentiles = ltv_forecasts[['ltv_p10', 'ltv_p25', 'ltv_median', 'ltv_p75', 'ltv_p90']].mean()
        percentile_labels = ['P10', 'P25', 'Median', 'P75', 'P90']
        bars = ax5.bar(percentile_labels, ltv_percentiles, color='mediumpurple', alpha=0.7)
        ax5.set_xlabel('Percentile')
        ax5.set_ylabel('LTV ($)')
        ax5.set_title('LTV Uncertainty Distribution')
        ax5.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, ltv_percentiles):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     f'${value:.0f}', ha='center', va='bottom')

        # 6. Cohort retention heatmap
        ax6 = axes[1, 2]
        cohort_data = self.prepare_cohort_analysis(df)

        # Create pivot table for heatmap
        retention_matrix = cohort_data.pivot_table(
            index='signup_cohort',
            columns='months_since_signup',
            values='retention_rate',
            fill_value=0
        )

        # Limit to first 24 months and last 12 cohorts for readability
        retention_matrix = retention_matrix.iloc[-12:, :24]

        sns.heatmap(retention_matrix, annot=False, cmap='YlOrRd', ax=ax6,
                    cbar_kws={'label': 'Retention Rate'})
        ax6.set_xlabel('Months Since Signup')
        ax6.set_ylabel('Signup Cohort')
        ax6.set_title('Cohort Retention Heatmap')

        # 7. Revenue trends by cohort
        ax7 = axes[2, 0]
        revenue_trends = cohort_data.pivot_table(
            index='months_since_signup',
            columns='signup_cohort',
            values='avg_revenue_per_customer',
            fill_value=0
        )

        # Plot trends for last 6 cohorts
        for cohort in revenue_trends.columns[-6:]:
            cohort_data_trend = revenue_trends[cohort][revenue_trends[cohort] > 0]
            if len(cohort_data_trend) > 1:
                ax7.plot(cohort_data_trend.index, cohort_data_trend.values,
                         marker='o', label=str(cohort), alpha=0.7)

        ax7.set_xlabel('Months Since Signup')
        ax7.set_ylabel('Average Revenue per Customer ($)')
        ax7.set_title('Revenue Trends by Signup Cohort')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)

        # 8. LTV/CAC analysis by channel
        ax8 = axes[2, 1]
        # Assume CAC (Customer Acquisition Cost) based on channel
        cac_by_channel = {
            'organic': 50, 'referral': 75, 'email': 100,
            'paid_search': 200, 'social': 150
        }

        channel_metrics = []
        for channel in ltv_forecasts['acquisition_channel'].unique():
            channel_ltv = ltv_forecasts[ltv_forecasts['acquisition_channel'] == channel]['ltv_median'].mean()
            channel_cac = cac_by_channel.get(channel, 100)
            ltv_cac_ratio = channel_ltv / channel_cac if channel_cac > 0 else 0

            channel_metrics.append({
                'channel': channel,
                'ltv': channel_ltv,
                'cac': channel_cac,
                'ltv_cac_ratio': ltv_cac_ratio
            })

        channel_metrics_df = pd.DataFrame(channel_metrics)

        # Create grouped bar chart
        x = np.arange(len(channel_metrics_df))
        width = 0.35

        bars1 = ax8.bar(x - width / 2, channel_metrics_df['ltv'], width,
                        label='LTV', color='lightblue', alpha=0.7)
        bars2 = ax8.bar(x + width / 2, channel_metrics_df['cac'], width,
                        label='CAC', color='lightcoral', alpha=0.7)

        ax8.set_xlabel('Acquisition Channel')
        ax8.set_ylabel('Value ($)')
        ax8.set_title('LTV vs CAC by Channel')
        ax8.set_xticks(x)
        ax8.set_xticklabels(channel_metrics_df['channel'], rotation=45)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Add ratio annotations
        for i, ratio in enumerate(channel_metrics_df['ltv_cac_ratio']):
            ax8.text(i, max(channel_metrics_df['ltv'].max(), channel_metrics_df['cac'].max()) * 1.1,
                     f'Ratio: {ratio:.1f}x', ha='center', va='bottom', fontweight='bold')

        # 9. Feature importance for revenue prediction
        ax9 = axes[2, 2]
        if 'random_forest' in self.feature_importance:
            importance_df = self.feature_importance['random_forest'].head(10)
            bars = ax9.barh(importance_df['feature'], importance_df['importance'],
                            color='gold', alpha=0.7)
            ax9.set_xlabel('Feature Importance')
            ax9.set_title('Top 10 Revenue Prediction Features')
            ax9.grid(True, alpha=0.3)

            # Clean up feature names for display
            feature_labels = [feat.replace('_', ' ').title() for feat in importance_df['feature']]
            ax9.set_yticklabels(feature_labels)

        plt.suptitle('Customer Lifetime Value Analysis Dashboard', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig('4ltv_forecast/ltv_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_pricing_recommendations(self, ltv_forecasts, pricing_analysis):
        """Generate actionable pricing recommendations based on LTV analysis"""

        recommendations = []

        # Overall LTV insights
        overall_ltv = ltv_forecasts['ltv_median'].mean()
        ltv_std = ltv_forecasts['ltv_median'].std()

        recommendations.append(f"""
🎯 EXECUTIVE SUMMARY - LTV INSIGHTS
=====================================
• Average Customer LTV: ${overall_ltv:.0f} (±${ltv_std:.0f})
• LTV Range: ${ltv_forecasts['ltv_median'].min():.0f} - ${ltv_forecasts['ltv_median'].max():.0f}
• High-value customers (P75+): ${ltv_forecasts['ltv_p75'].mean():.0f} LTV
• Risk-adjusted LTV (P25): ${ltv_forecasts['ltv_p25'].mean():.0f}
""")

        # Channel-specific insights
        channel_ltv = ltv_forecasts.groupby('acquisition_channel')['ltv_median'].mean().sort_values(ascending=False)

        recommendations.append(f"""
📊 ACQUISITION CHANNEL OPTIMIZATION
===================================
HIGHEST VALUE CHANNELS:
• {channel_ltv.index[0]}: ${channel_ltv.iloc[0]:.0f} average LTV
• {channel_ltv.index[1]}: ${channel_ltv.iloc[1]:.0f} average LTV

LOWEST VALUE CHANNELS:
• {channel_ltv.index[-1]}: ${channel_ltv.iloc[-1]:.0f} average LTV
• {channel_ltv.index[-2]}: ${channel_ltv.iloc[-2]:.0f} average LTV

RECOMMENDATIONS:
→ Increase investment in {channel_ltv.index[0]} and {channel_ltv.index[1]}
→ Improve conversion quality for {channel_ltv.index[-1]} channel
→ Consider raising CAC limits for top-performing channels
""")

        # Plan-specific insights
        plan_ltv = ltv_forecasts.groupby('plan_type')['ltv_median'].mean().sort_values(ascending=False)
        plan_counts = ltv_forecasts['plan_type'].value_counts()

        recommendations.append(f"""
💰 PRICING STRATEGY RECOMMENDATIONS
==================================
PLAN PERFORMANCE:
• {plan_ltv.index[0]}: ${plan_ltv.iloc[0]:.0f} LTV ({plan_counts[plan_ltv.index[0]]} customers)
• {plan_ltv.index[1]}: ${plan_ltv.iloc[1]:.0f} LTV ({plan_counts[plan_ltv.index[1]]} customers)
• {plan_ltv.index[2]}: ${plan_ltv.iloc[2]:.0f} LTV ({plan_counts[plan_ltv.index[2]]} customers)

PRICING ACTIONS:
1. UPSELLING OPPORTUNITY: {plan_counts.index[0]} plan has {plan_counts.iloc[0]} customers
   → Target for upgrades to higher-value plans

2. VALUE OPTIMIZATION: {plan_ltv.index[0]} plan shows highest LTV
   → Consider premium features or pricing tiers

3. CONVERSION FOCUS: Improve free-to-paid conversion
   → Current free user LTV suggests strong upgrade potential
""")

        # Uncertainty and risk management
        ltv_cv = ltv_forecasts['ltv_std'] / ltv_forecasts['ltv_mean']  # Coefficient of variation
        high_uncertainty_customers = ltv_forecasts[ltv_cv > ltv_cv.quantile(0.8)]

        recommendations.append(f"""
⚠️ RISK MANAGEMENT & UNCERTAINTY
================================
UNCERTAINTY ANALYSIS:
• {len(high_uncertainty_customers)} customers ({len(high_uncertainty_customers) / len(ltv_forecasts) * 100:.1f}%) 
  have high LTV uncertainty (CV > 80th percentile)
• Average uncertainty: ±{ltv_forecasts['ltv_std'].mean():.0f} around mean LTV

RISK MITIGATION STRATEGIES:
1. PORTFOLIO APPROACH: Diversify customer acquisition across channels
2. EARLY WARNING SYSTEM: Monitor customers with high uncertainty scores
3. RETENTION PROGRAMS: Focus on customers with high potential but high risk
4. CONSERVATIVE PLANNING: Use P25 LTV (${ltv_forecasts['ltv_p25'].mean():.0f}) for financial planning
""")

        # Pricing optimization based on analysis
        if pricing_analysis:
            recommendations.append(f"""
🔧 TACTICAL PRICING RECOMMENDATIONS
===================================
""")

            for plan_type, analysis_df in pricing_analysis.items():
                analysis_df.dropna(inplace=True)
                if analysis_df.shape[0] < 1:
                    continue
                optimal_row = analysis_df.loc[analysis_df['total_value_index'].idxmax()]
                current_performance = analysis_df.loc[analysis_df['price_multiplier'].sub(1.0).abs().idxmin()]

                recommendations.append(f"""
{plan_type.upper()} PLAN:
• Current Performance: {current_performance['total_value_index']:.2f}x value index
• Optimal Price Point: {optimal_row['price_multiplier']:.1f}x current price
• Expected Volume Impact: {optimal_row['demand_change_pct']:.1f}%
• Value Improvement: {optimal_row['total_value_index']:.2f}x
→ {"INCREASE" if optimal_row['price_multiplier'] > 1.0 else "DECREASE"} price by {abs(optimal_row['price_multiplier'] - 1) * 100:.0f}%
""")

        # Cohort-specific strategies
        cohort_performance = ltv_forecasts.groupby('company_size')['ltv_median'].mean().sort_values(ascending=False)

        recommendations.append(f"""
🏢 SEGMENT-SPECIFIC STRATEGIES
=============================
COMPANY SIZE OPTIMIZATION:
• Enterprise (1000+): ${cohort_performance.get('1000+', 0):.0f} LTV → Premium pricing justified
• Mid-market (201-1000): ${cohort_performance.get('201-1000', 0):.0f} LTV → Volume pricing strategy
• SMB (1-50): ${cohort_performance.get('1-10', 0):.0f} LTV → Efficiency-focused pricing

INDUSTRY TARGETING:
""")

        industry_ltv = ltv_forecasts.groupby('industry')['ltv_median'].mean().sort_values(ascending=False)
        for industry in industry_ltv.index[:3]:
            recommendations.append(f"• {industry.title()}: ${industry_ltv[industry]:.0f} LTV - High priority segment")

        recommendations.append(f"""

📈 IMPLEMENTATION ROADMAP
========================
IMMEDIATE (0-30 days):
• Implement LTV tracking dashboard
• Segment customers by uncertainty score
• A/B test pricing on low-risk segments

SHORT-TERM (1-3 months):
• Launch retention programs for high-uncertainty customers
• Optimize acquisition spend by channel LTV
• Implement predictive churn scoring

LONG-TERM (3-6 months):
• Dynamic pricing based on customer characteristics
• Personalized upgrade recommendations
• Advanced cohort-based forecasting

💡 KEY SUCCESS METRICS
=====================
• LTV/CAC ratio > 3:1 for all channels
• Customer payback period < 12 months
• LTV prediction accuracy > 85%
• Churn rate reduction of 15%
""")

        return '\n'.join(recommendations)


# # Example usage
def main():
    # Initialize the LTV forecaster
    ltv_forecaster = CustomerLTVForecaster()

    """Run the complete LTV forecasting pipeline"""
    n_customers = 10000
    print("🚀 Starting Customer LTV Forecasting Analysis...")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n📊 Generating synthetic customer data...")
    df = ltv_forecaster.generate_synthetic_customer_data(n_customers=n_customers)
    print(f"✅ Generated {len(df)} customer-month records for {df['customer_id'].nunique()} customers")

    # 2. Fit survival models
    print("\n⏱️ Fitting survival models...")
    survival_data = ltv_forecaster.fit_survival_models(df)
    print(f"✅ Fitted survival models on {len(survival_data)} customers")

    # 3. Fit revenue models
    print("\n💰 Fitting revenue prediction models...")
    revenue_data = ltv_forecaster.fit_revenue_models(df)
    print(f"✅ Trained revenue models on {len(revenue_data)} observations")

    # 4. Generate LTV forecasts
    print("\n🔮 Generating probabilistic LTV forecasts...")
    ltv_forecasts = ltv_forecaster.forecast_ltv_probabilistic(df, forecast_months=6)
    print(f"✅ Generated LTV forecasts for {len(ltv_forecasts)} customers")

    # 5. Cohort analysis
    print("\n👥 Calculating cohort-based LTV analysis...")
    cohort_analysis = ltv_forecaster.calculate_cohort_ltv(ltv_forecasts)
    print("✅ Completed cohort analysis")

    # 6. Pricing sensitivity analysis
    print("\n💵 Running pricing sensitivity analysis...")
    current_prices = {'free': 0, 'basic': 29, 'pro': 99, 'enterprise': 399}
    pricing_analysis = ltv_forecaster.pricing_sensitivity_analysis(ltv_forecasts, current_prices)
    print("✅ Completed pricing analysis")

    # 7. Generate visualizations
    print("\n📈 Creating analysis dashboard...")
    ltv_forecaster.plot_ltv_analysis(df, ltv_forecasts, cohort_analysis)
    print("✅ Dashboard created and saved")

    # 8. Generate recommendations
    print("\n📋 Generating strategic recommendations...")
    recommendations = ltv_forecaster.generate_pricing_recommendations(ltv_forecasts, pricing_analysis)

    # 9. Summary statistics
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"""
    CUSTOMER BASE OVERVIEW:
    • Total Customers Analyzed: {ltv_forecasts['customer_id'].nunique():,}
    • Average Customer LTV: ${ltv_forecasts['ltv_median'].mean():.2f}
    • LTV Standard Deviation: ${ltv_forecasts['ltv_median'].std():.2f}
    • Median Customer Survival (12m): {ltv_forecasts['survival_prob_12m'].mean():.1%}
    • Median Customer Survival (24m): {ltv_forecasts['survival_prob_24m'].mean():.1%}
    
    TOP PERFORMING SEGMENTS:
    • Best Acquisition Channel: {ltv_forecasts.groupby('acquisition_channel')['ltv_median'].mean().idxmax()}
    • Best Plan Type: {ltv_forecasts.groupby('plan_type')['ltv_median'].mean().idxmax()}
    • Best Company Size: {ltv_forecasts.groupby('company_size')['ltv_median'].mean().idxmax()}
    
    UNCERTAINTY METRICS:
    • Coefficient of Variation: {(ltv_forecasts['ltv_std'] / ltv_forecasts['ltv_mean']).mean():.2f}
    • P10-P90 Range: ${ltv_forecasts['ltv_p10'].mean():.0f} - ${ltv_forecasts['ltv_p90'].mean():.0f}
    """)

    print(recommendations)
    results = {
        'data': df,
        'ltv_forecasts': ltv_forecasts,
        'cohort_analysis': cohort_analysis,
        'pricing_analysis': pricing_analysis,
        'recommendations': recommendations
    }

    # Access specific results
    ltv_data = results['ltv_forecasts']
    pricing_recommendations = results['recommendations']

    # Export key results
    ltv_data.to_csv('4ltv_forecast/customer_ltv_forecasts.csv', index=False)

    with open('4ltv_forecast/ltv_pricing_recommendations.txt', 'w') as f:
        f.write(pricing_recommendations)

    print("\n✅ Analysis complete! Results exported to CSV and text files.")


# # Example usage
if __name__ == "__main__":
    main()
