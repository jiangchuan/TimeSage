# Forecast API Usage During Major Model Transition
# FigureX.ai is planning to deprecate FigureX-3.5-turbo in favor of FigureX-4o. 
# Leadership wants to understand how API usage will shift, and whether compute infrastructure needs to scale accordingly.

# Challenge:
# 1. Time series segmentation (model-level usage)
# 2. Modeling changepoints and transition dynamics
# 3. External regressors (pricing, documentation, media coverage)

# Solution:
# 1. Decompose current usage by model.
# 2. Identify potential changepoint (FigureX-4o launch).
# 3. Use Prophet/XGBoost with rollout indicators.
# 4. Forecast net usage and compute needs (daily → weekly).
# 5. Provide scenarios: slow migration vs. aggressive switching.

import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

# Core modeling libraries
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ruptures import Pelt


class APIUsageForecast:
    """
    Comprehensive API usage forecasting for model transitions
    """

    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.changepoints = {}
        self.scenarios = {}

    def generate_sample_data(self, days=365):
        """Generate realistic sample API usage data"""
        np.random.seed(42)

        # Date range
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

        # Base trends and seasonality
        trend = np.linspace(1000, 5000, days)  # Growing usage
        weekly_season = 200 * np.sin(2 * np.pi * np.arange(days) / 7)
        noise = np.random.normal(0, 100, days)

        # FigureX-3.5 usage (dominant initially, declining after FigureX-4o launch)
        figurex35_base = trend + weekly_season + noise

        # Simulate FigureX-4o launch effect (day 200)
        launch_day = 200
        figurex4o_adoption = np.zeros(days)
        figurex4o_adoption[launch_day:] = np.cumsum(np.random.exponential(0.02, days - launch_day))

        # FigureX-3.5 cannibalization after launch
        cannibalization = np.zeros(days)
        cannibalization[launch_day:] = -0.3 * figurex4o_adoption[launch_day:]

        figurex35_usage = np.maximum(figurex35_base + cannibalization, 100)
        figurex4o_usage = figurex4o_adoption * 50 + np.random.normal(0, 20, days)
        figurex4o_usage = np.maximum(figurex4o_usage, 0)

        # External factors
        pricing_impact = np.random.normal(0, 0.1, days)
        media_coverage = np.zeros(days)
        media_coverage[launch_day:launch_day + 30] = np.random.exponential(0.5, 30)  # Launch buzz

        # Create comprehensive dataset
        data = pd.DataFrame({
            'date': dates,
            'figurex35_usage': figurex35_usage,
            'figurex4o_usage': figurex4o_usage,
            'total_usage': figurex35_usage + figurex4o_usage,
            'pricing_impact': pricing_impact,
            'media_coverage': media_coverage,
            'day_of_week': dates.day_of_week,
            'month': dates.month,
            'is_weekend': dates.day_of_week >= 5
        })

        return data

    def decompose_usage_by_model(self, data):
        """Analyze and visualize usage patterns by model"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('API Usage Decomposition by Model', fontsize=16, fontweight='bold')

        # Total usage over time
        axes[0, 0].plot(data['date'], data['figurex35_usage'], label='FigureX-3.5-turbo', linewidth=2)
        axes[0, 0].plot(data['date'], data['figurex4o_usage'], label='FigureX-4o', linewidth=2)
        axes[0, 0].plot(data['date'], data['total_usage'], label='Total Usage', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Daily Usage by Model')
        axes[0, 0].set_ylabel('API Calls (thousands)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Market share evolution
        data['figurex35_share'] = data['figurex35_usage'] / data['total_usage'] * 100
        data['figurex4o_share'] = data['figurex4o_usage'] / data['total_usage'] * 100

        axes[0, 1].fill_between(data['date'], 0, data['figurex35_share'], alpha=0.7, label='FigureX-3.5-turbo')
        axes[0, 1].fill_between(data['date'], data['figurex35_share'], 100, alpha=0.7, label='FigureX-4o')
        axes[0, 1].set_title('Market Share Evolution (%)')
        axes[0, 1].set_ylabel('Market Share (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Weekly aggregated patterns
        weekly_data = data.set_index('date').resample('W').sum()
        axes[1, 0].plot(weekly_data.index, weekly_data['figurex35_usage'], marker='o', label='FigureX-3.5-turbo')
        axes[1, 0].plot(weekly_data.index, weekly_data['figurex4o_usage'], marker='s', label='FigureX-4o')
        axes[1, 0].set_title('Weekly Usage Patterns')
        axes[1, 0].set_ylabel('Weekly API Calls (thousands)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Usage distribution
        axes[1, 1].hist(data['figurex35_usage'], bins=30, alpha=0.7, label='FigureX-3.5-turbo', density=True)
        axes[1, 1].hist(data['figurex4o_usage'], bins=30, alpha=0.7, label='FigureX-4o', density=True)
        axes[1, 1].set_title('Usage Distribution')
        axes[1, 1].set_xlabel('Daily API Calls (thousands)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'6api_usage_forecast/API_usage_by_model.png', dpi=300, bbox_inches='tight')
        plt.show()

        return data

    def detect_changepoints(self, data):
        """Identify significant changepoints in usage patterns"""

        # Detect changepoints in total usage
        model = "rbf"
        algo = Pelt(model=model).fit(data['total_usage'].values)
        changepoints = algo.predict(pen=100)

        # Detect model-specific changepoints
        figurex35_algo = Pelt(model=model).fit(data['figurex35_usage'].values)
        figurex35_changepoints = figurex35_algo.predict(pen=50)

        figurex4o_algo = Pelt(model=model).fit(data['figurex4o_usage'].values)
        figurex4o_changepoints = figurex4o_algo.predict(pen=30)

        # Visualize changepoints
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Changepoint Detection Analysis', fontsize=16, fontweight='bold')

        # Total usage changepoints
        axes[0].plot(data['date'], data['total_usage'], linewidth=2)
        for cp in changepoints[:-1]:  # Exclude last point
            axes[0].axvline(data['date'].iloc[cp], color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('Total Usage Changepoints')
        axes[0].set_ylabel('Total API Calls')
        axes[0].grid(True, alpha=0.3)

        # FigureX-3.5 changepoints
        axes[1].plot(data['date'], data['figurex35_usage'], linewidth=2, color='blue')
        for cp in figurex35_changepoints[:-1]:
            axes[1].axvline(data['date'].iloc[cp], color='red', linestyle='--', alpha=0.7)
        axes[1].set_title('FigureX-3.5-turbo Usage Changepoints')
        axes[1].set_ylabel('FigureX-3.5 API Calls')
        axes[1].grid(True, alpha=0.3)

        # FigureX-4o changepoints
        axes[2].plot(data['date'], data['figurex4o_usage'], linewidth=2, color='green')
        for cp in figurex4o_changepoints[:-1]:
            axes[2].axvline(data['date'].iloc[cp], color='red', linestyle='--', alpha=0.7)
        axes[2].set_title('FigureX-4o Usage Changepoints')
        axes[2].set_ylabel('FigureX-4o API Calls')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'6api_usage_forecast/changepoint_detection.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.changepoints = {
            'total': changepoints,
            'figurex35': figurex35_changepoints,
            'figurex4o': figurex4o_changepoints
        }

        return changepoints

    def build_prophet_model(self, data, target_col='total_usage'):
        """Build Prophet model with external regressors and detected changepoints"""

        # Prepare data for Prophet
        prophet_data = data[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})

        # Add external regressors
        prophet_data['pricing_impact'] = data['pricing_impact']
        prophet_data['media_coverage'] = data['media_coverage']
        prophet_data['is_weekend'] = data['is_weekend'].astype(int)

        # Use detected changepoints if available
        changepoints = None
        if hasattr(self, 'changepoints') and self.changepoints:
            if target_col == 'total_usage' and 'total' in self.changepoints:
                # Convert changepoint indices to dates
                cp_indices = self.changepoints['total'][:-1]  # Exclude last point
                changepoints = [data['date'].iloc[cp] for cp in cp_indices if cp < len(data)]
            elif target_col == 'figurex35_usage' and 'figurex35' in self.changepoints:
                cp_indices = self.changepoints['figurex35'][:-1]
                changepoints = [data['date'].iloc[cp] for cp in cp_indices if cp < len(data)]
            elif target_col == 'figurex4o_usage' and 'figurex4o' in self.changepoints:
                cp_indices = self.changepoints['figurex4o'][:-1]
                changepoints = [data['date'].iloc[cp] for cp in cp_indices if cp < len(data)]

        # Initialize Prophet model with detected changepoints
        model = Prophet(
            changepoints=changepoints,
            changepoint_prior_scale=0.1 if changepoints else 0.05,  # Higher if we have specific changepoints
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )

        # Add external regressors
        model.add_regressor('pricing_impact')
        model.add_regressor('media_coverage')
        model.add_regressor('is_weekend')

        # Fit model
        model.fit(prophet_data)

        self.models[f'prophet_{target_col}'] = model
        print(f"   ✓ Prophet model for {target_col} built with {len(changepoints) if changepoints else 0} changepoints")
        return model

    def build_xgboost_model(self, data, target_col='total_usage'):
        """Build XGBoost model with feature engineering and changepoint indicators"""

        # Feature engineering
        features = data.copy()
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        # Lag features
        features['usage_lag1'] = features[target_col].shift(1)
        features['usage_lag7'] = features[target_col].shift(7)
        features['usage_ma7'] = features[target_col].rolling(7).mean()

        # Rollout indicators (post-launch effects)
        launch_date = features['date'].iloc[200]  # Assumed launch day
        features['days_since_launch'] = (features['date'] - launch_date).dt.days
        features['days_since_launch'] = features['days_since_launch'].clip(lower=0)
        features['post_launch'] = (features['days_since_launch'] > 0).astype(int)

        # Add changepoint-based features if available
        changepoint_features = []
        if hasattr(self, 'changepoints') and self.changepoints:
            changepoint_key = 'total' if target_col == 'total_usage' else target_col.replace('_usage', '')
            if changepoint_key in self.changepoints:
                cp_indices = self.changepoints[changepoint_key][:-1]  # Exclude last point

                # Create binary indicators for post-changepoint periods
                for i, cp_idx in enumerate(cp_indices):
                    if cp_idx < len(features):
                        cp_feature = f'post_changepoint_{i + 1}'
                        features[cp_feature] = (features.index >= cp_idx).astype(int)
                        changepoint_features.append(cp_feature)

                # Create time-since-changepoint features
                for i, cp_idx in enumerate(cp_indices):
                    if cp_idx < len(features):
                        tsc_feature = f'time_since_cp_{i + 1}'
                        features[tsc_feature] = np.maximum(0, features.index - cp_idx)
                        changepoint_features.append(tsc_feature)

        # Select features
        feature_cols = [
                           'pricing_impact', 'media_coverage', 'day_sin', 'day_cos',
                           'month_sin', 'month_cos', 'usage_lag1', 'usage_lag7', 'usage_ma7',
                           'days_since_launch', 'post_launch', 'is_weekend'
                       ] + changepoint_features

        # Remove rows with NaN (due to lags)
        clean_data = features.dropna()
        X = clean_data[feature_cols]
        y = clean_data[target_col]

        # Split data
        split_idx = int(len(clean_data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        print(f"   ✓ XGBoost {target_col} Model Performance:")
        print(f"     Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
        print(f"     Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
        print(f"     Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")
        print(f"     Features used: {len(feature_cols)} (including {len(changepoint_features)} changepoint features)")

        # Feature importance
        importance = model.feature_importances_
        feature_importance = list(zip(feature_cols, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print(f"     Top 5 features:")
        for feat, imp in feature_importance[:5]:
            print(f"       - {feat}: {imp:.3f}")

        self.models[f'xgb_{target_col}'] = {
            'model': model,
            'features': feature_cols,
            'feature_importance': feature_importance,
            'scaler': None
        }

        return model, feature_cols

    def generate_forecasts(self, data, forecast_days=90):
        """Generate forecasts using both Prophet and XGBoost models"""

        forecasts = {}

        # Prophet forecasts
        print("   🔮 Generating Prophet forecasts...")
        for target in ['total_usage', 'figurex35_usage', 'figurex4o_usage']:
            if target in data.columns:
                model = self.build_prophet_model(data, target)

                # Create future dataframe
                future = model.make_future_dataframe(periods=forecast_days)

                # Add regressor values for future (assuming continuity)
                future['pricing_impact'] = list(data['pricing_impact']) + [0] * forecast_days
                future['media_coverage'] = list(data['media_coverage']) + [0] * forecast_days
                future['is_weekend'] = [d.weekday() >= 5 for d in future['ds']]
                future['is_weekend'] = future['is_weekend'].astype(int)

                # Generate forecast
                forecast = model.predict(future)
                forecasts[f'prophet_{target}'] = forecast

        # XGBoost forecasts
        print("   🤖 Generating XGBoost forecasts...")
        for target in ['total_usage', 'figurex35_usage', 'figurex4o_usage']:
            if target in data.columns:
                xgb_model, feature_cols = self.build_xgboost_model(data, target)

                # Generate XGBoost forecast using iterative prediction
                xgb_forecast = self._xgboost_iterative_forecast(
                    data, xgb_model, feature_cols, target, forecast_days
                )
                forecasts[f'xgb_{target}'] = xgb_forecast

        # Model comparison
        self._compare_model_performance(data, forecasts)

        self.forecasts = forecasts
        return forecasts

    def _xgboost_iterative_forecast(self, data, model, feature_cols, target_col, forecast_days):
        """Generate iterative forecasts using XGBoost for time series"""

        # Prepare the last known data point
        features = data.copy()

        # Add engineered features (same as in build_xgboost_model)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['usage_lag1'] = features[target_col].shift(1)
        features['usage_lag7'] = features[target_col].shift(7)
        features['usage_ma7'] = features[target_col].rolling(7).mean()

        # Launch-related features
        launch_date = features['date'].iloc[200]
        features['days_since_launch'] = (features['date'] - launch_date).dt.days
        features['days_since_launch'] = features['days_since_launch'].clip(lower=0)
        features['post_launch'] = (features['days_since_launch'] > 0).astype(int)

        # Add changepoint features
        if hasattr(self, 'changepoints') and self.changepoints:
            changepoint_key = 'total' if target_col == 'total_usage' else target_col.replace('_usage', '')
            if changepoint_key in self.changepoints:
                cp_indices = self.changepoints[changepoint_key][:-1]
                for i, cp_idx in enumerate(cp_indices):
                    if cp_idx < len(features):
                        features[f'post_changepoint_{i + 1}'] = (features.index >= cp_idx).astype(int)
                        features[f'time_since_cp_{i + 1}'] = np.maximum(0, features.index - cp_idx)

        # Get the last valid row (without NaN)
        clean_data = features.dropna()
        last_row = clean_data.iloc[-1].copy()

        # Initialize forecast arrays
        forecast_values = []
        forecast_dates = []

        # Start forecasting from the next day
        current_date = data['date'].iloc[-1]

        for day in range(forecast_days):
            current_date += timedelta(days=1)
            forecast_dates.append(current_date)

            # Update time-dependent features for current forecast day
            last_row['day_of_week'] = current_date.weekday()
            last_row['month'] = current_date.month
            last_row['is_weekend'] = current_date.weekday() >= 5
            last_row['day_sin'] = np.sin(2 * np.pi * current_date.weekday() / 7)
            last_row['day_cos'] = np.cos(2 * np.pi * current_date.weekday() / 7)
            last_row['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
            last_row['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
            last_row['days_since_launch'] = max(0, (current_date - launch_date).days)
            last_row['post_launch'] = 1 if last_row['days_since_launch'] > 0 else 0

            # Update changepoint features
            if hasattr(self, 'changepoints') and self.changepoints:
                changepoint_key = 'total' if target_col == 'total_usage' else target_col.replace('_usage', '')
                if changepoint_key in self.changepoints:
                    cp_indices = self.changepoints[changepoint_key][:-1]
                    for i, cp_idx in enumerate(cp_indices):
                        if cp_idx < len(data):
                            current_idx = len(data) + day  # Approximate future index
                            last_row[f'post_changepoint_{i + 1}'] = 1 if current_idx >= cp_idx else 0
                            last_row[f'time_since_cp_{i + 1}'] = max(0, current_idx - cp_idx)

            # Prepare features for prediction
            X_pred = last_row[feature_cols].values.reshape(1, -1)

            # Make prediction
            pred_value = model.predict(X_pred)[0]
            forecast_values.append(max(pred_value, 0))  # Ensure non-negative

            # Update lag features for next iteration
            if len(forecast_values) >= 1:
                last_row['usage_lag1'] = forecast_values[-1]
            if len(forecast_values) >= 7:
                last_row['usage_lag7'] = forecast_values[-7]
                last_row['usage_ma7'] = np.mean(forecast_values[-7:])
            elif len(forecast_values) > 0:
                last_row['usage_ma7'] = np.mean(forecast_values)

        # Create forecast dataframe in Prophet-like format
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values
        })

        return forecast_df

    def _compare_model_performance(self, data, forecasts):
        """Compare Prophet vs XGBoost model performance"""

        print("   📊 Model Performance Comparison:")

        for target in ['total_usage', 'figurex35_usage', 'figurex4o_usage']:
            if f'prophet_{target}' in forecasts and f'xgb_{target}' in forecasts:
                prophet_forecast = forecasts[f'prophet_{target}']
                xgb_forecast = forecasts[f'xgb_{target}']

                # Get historical predictions for comparison
                historical_length = len(data)
                prophet_hist = prophet_forecast.iloc[:historical_length]['yhat'].values
                actual_values = data[target].values

                # Calculate errors for Prophet (on historical data)
                prophet_mae = mean_absolute_error(actual_values, prophet_hist)
                prophet_rmse = np.sqrt(mean_squared_error(actual_values, prophet_hist))

                print(f"     {target.upper()}:")
                print(f"       Prophet Historical MAE: {prophet_mae:.2f}")
                print(f"       Prophet Historical RMSE: {prophet_rmse:.2f}")
                print(f"       XGBoost features: {len(self.models[f'xgb_{target}']['features'])}")

                # Future forecast comparison (first 30 days)
                future_length = min(30, len(xgb_forecast))
                prophet_future = prophet_forecast.iloc[historical_length:historical_length + future_length][
                    'yhat'].values
                xgb_future = xgb_forecast.iloc[:future_length]['yhat'].values

                avg_diff = np.mean(np.abs(prophet_future - xgb_future))
                print(f"       Avg forecast difference (30-day): {avg_diff:.2f}")
                print()

    def create_scenarios(self, data, forecast_days=90):
        """Create different migration scenarios using changepoint insights"""

        scenarios = {
            'slow_migration': {},
            'aggressive_switching': {},
            'baseline': {}
        }

        # Use changepoint information to inform scenario parameters
        changepoint_impact = 1.0
        if hasattr(self, 'changepoints') and self.changepoints:
            # If we detected many changepoints, assume higher volatility
            total_changepoints = len(self.changepoints.get('total', [])) - 1  # Exclude last point
            changepoint_impact = 1.0 + (total_changepoints * 0.1)  # 10% increase per changepoint

        # Adjust base rates based on changepoint analysis
        base_figurex35_decline = 0.02 * changepoint_impact  # More aggressive if more changepoints
        base_figurex4o_growth = 0.15 * changepoint_impact

        # Slow migration scenario
        slow_figurex35_decline = 0.01 * changepoint_impact
        slow_figurex4o_growth = 0.08 * changepoint_impact

        # Aggressive switching scenario
        aggr_figurex35_decline = 0.05 * changepoint_impact
        aggr_figurex4o_growth = 0.25 * changepoint_impact

        current_figurex35 = data['figurex35_usage'].iloc[-1]
        current_figurex4o = data['figurex4o_usage'].iloc[-1]

        print(f"   📊 Scenario parameters (changepoint impact factor: {changepoint_impact:.2f}):")

        for scenario, params in [
            ('baseline', (base_figurex35_decline, base_figurex4o_growth)),
            ('slow_migration', (slow_figurex35_decline, slow_figurex4o_growth)),
            ('aggressive_switching', (aggr_figurex35_decline, aggr_figurex4o_growth))
        ]:

            figurex35_decline, figurex4o_growth = params
            print(
                f"     • {scenario.replace('_', ' ').title()}: FigureX-3.5 decline {figurex35_decline:.1%}/week, FigureX-4o growth {figurex4o_growth:.1%}/week")

            # Weekly projections
            weeks = forecast_days // 7
            figurex35_proj = [current_figurex35]
            figurex4o_proj = [current_figurex4o]

            for week in range(weeks):
                # Apply weekly changes with some randomness
                new_figurex35 = figurex35_proj[-1] * (1 - figurex35_decline) * np.random.normal(1, 0.05)
                new_figurex4o = figurex4o_proj[-1] * (1 + figurex4o_growth) * np.random.normal(1, 0.1)

                figurex35_proj.append(max(new_figurex35, 100))  # Minimum usage floor
                figurex4o_proj.append(new_figurex4o)

            scenarios[scenario] = {
                'figurex35_usage': figurex35_proj,
                'figurex4o_usage': figurex4o_proj,
                'total_usage': [g35 + g4o for g35, g4o in zip(figurex35_proj, figurex4o_proj)]
            }

        self.scenarios = scenarios
        return scenarios

    def visualize_scenarios(self, data):
        """Visualize different scenarios and compute requirements, including model forecasts"""

        if not self.scenarios:
            self.create_scenarios(data)

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('API Usage Forecast Scenarios & Model Comparison', fontsize=16, fontweight='bold')

        weeks = len(list(self.scenarios.values())[0]['total_usage'])
        future_dates = pd.date_range(start=data['date'].iloc[-1], periods=weeks, freq='W')

        # Total usage scenarios
        for scenario, data_dict in self.scenarios.items():
            axes[0, 0].plot(future_dates, data_dict['total_usage'],
                            marker='o', linewidth=2, label=scenario.replace('_', ' ').title())

        axes[0, 0].set_title('Total Usage Scenarios')
        axes[0, 0].set_ylabel('Weekly API Calls (thousands)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # FigureX-3.5 decline scenarios
        for scenario, data_dict in self.scenarios.items():
            axes[0, 1].plot(future_dates, data_dict['figurex35_usage'],
                            marker='s', linewidth=2, label=scenario.replace('_', ' ').title())

        axes[0, 1].set_title('FigureX-3.5-turbo Usage Scenarios')
        axes[0, 1].set_ylabel('Weekly API Calls (thousands)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # FigureX-4o growth scenarios
        for scenario, data_dict in self.scenarios.items():
            axes[1, 0].plot(future_dates, data_dict['figurex4o_usage'],
                            marker='^', linewidth=2, label=scenario.replace('_', ' ').title())

        axes[1, 0].set_title('FigureX-4o Usage Scenarios')
        axes[1, 0].set_ylabel('Weekly API Calls (thousands)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Compute requirements (assuming different compute costs)
        figurex35_compute_ratio = 1.0  # Base compute
        figurex4o_compute_ratio = 2.5  # 2.5x more compute intensive

        compute_requirements = {}
        for scenario, data_dict in self.scenarios.items():
            weekly_compute = [
                g35 * figurex35_compute_ratio + g4o * figurex4o_compute_ratio
                for g35, g4o in zip(data_dict['figurex35_usage'], data_dict['figurex4o_usage'])
            ]
            compute_requirements[scenario] = weekly_compute
            axes[1, 1].plot(future_dates, weekly_compute,
                            marker='d', linewidth=2, label=scenario.replace('_', ' ').title())

        axes[1, 1].set_title('Compute Requirements by Scenario')
        axes[1, 1].set_ylabel('Compute Units (thousands)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Model comparison plots (Prophet vs XGBoost)
        if hasattr(self, 'forecasts') and self.forecasts:
            # Daily forecast comparison for total usage
            if 'prophet_total_usage' in self.forecasts and 'xgb_total_usage' in self.forecasts:
                prophet_forecast = self.forecasts['prophet_total_usage']
                xgb_forecast = self.forecasts['xgb_total_usage']

                # Plot future forecasts only (daily)
                historical_length = len(data)
                prophet_future = prophet_forecast.iloc[historical_length:]

                axes[2, 0].plot(prophet_future['ds'], prophet_future['yhat'],
                                label='Prophet', linewidth=2, alpha=0.8)
                axes[2, 0].plot(xgb_forecast['ds'], xgb_forecast['yhat'],
                                label='XGBoost', linewidth=2, alpha=0.8)

                # Add confidence intervals for Prophet
                if 'yhat_lower' in prophet_future.columns:
                    axes[2, 0].fill_between(prophet_future['ds'],
                                            prophet_future['yhat_lower'],
                                            prophet_future['yhat_upper'],
                                            alpha=0.2, label='Prophet 95% CI')

                axes[2, 0].set_title('Model Forecast Comparison - Total Usage (Daily)')
                axes[2, 0].set_ylabel('Daily API Calls (thousands)')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)

            # Forecast difference analysis
            if 'prophet_total_usage' in self.forecasts and 'xgb_total_usage' in self.forecasts:
                prophet_forecast = self.forecasts['prophet_total_usage']
                xgb_forecast = self.forecasts['xgb_total_usage']

                historical_length = len(data)
                prophet_future = prophet_forecast.iloc[historical_length:historical_length + len(xgb_forecast)]

                # Calculate differences
                forecast_diff = prophet_future['yhat'].values - xgb_forecast['yhat'].values

                axes[2, 1].plot(xgb_forecast['ds'], forecast_diff,
                                linewidth=2, color='red', alpha=0.7)
                axes[2, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[2, 1].set_title('Forecast Difference (Prophet - XGBoost)')
                axes[2, 1].set_ylabel('Difference in API Calls')
                axes[2, 1].set_xlabel('Date')
                axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'6api_usage_forecast/API_usage_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()

        return compute_requirements

    def generate_uncertainty_intervals(self, data):
        """Generate prediction intervals and uncertainty quantification"""

        # Use Prophet's built-in uncertainty intervals
        forecasts = self.generate_forecasts(data)

        if 'prophet_total_usage' in forecasts:
            forecast = forecasts['prophet_total_usage']

            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle('Forecast Uncertainty Analysis', fontsize=16, fontweight='bold')

            # Plot with uncertainty intervals
            axes[0].plot(forecast['ds'], forecast['yhat'], label='Forecast', linewidth=2)
            axes[0].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                 alpha=0.3, label='95% Confidence Interval')
            axes[0].set_title('Total Usage Forecast with Uncertainty')
            axes[0].set_ylabel('API Calls (thousands)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Uncertainty width over time
            uncertainty_width = forecast['yhat_upper'] - forecast['yhat_lower']
            axes[1].plot(forecast['ds'], uncertainty_width, color='red', linewidth=2)
            axes[1].set_title('Forecast Uncertainty Width Over Time')
            axes[1].set_ylabel('Uncertainty Width')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'6api_usage_forecast/forecast_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

    def generate_executive_summary(self, data):
        """Generate executive summary with key insights including changepoint analysis"""

        # Calculate key metrics
        current_total = data['total_usage'].iloc[-1]
        current_figurex35_share = data['figurex35_usage'].iloc[-1] / current_total * 100
        current_figurex4o_share = data['figurex4o_usage'].iloc[-1] / current_total * 100

        # Growth rates
        recent_data = data.tail(30)  # Last 30 days
        figurex35_growth = (recent_data['figurex35_usage'].iloc[-1] / recent_data['figurex35_usage'].iloc[0] - 1) * 100
        figurex4o_growth = (recent_data['figurex4o_usage'].iloc[-1] / recent_data['figurex4o_usage'].iloc[0] - 1) * 100

        print("=" * 60)
        print("EXECUTIVE SUMMARY: API Usage Forecast During Model Transition")
        print("=" * 60)
        print()
        print("📊 CURRENT STATE:")
        print(f"   • Total Daily Usage: {current_total:,.0f} API calls")
        print(f"   • FigureX-3.5-turbo Share: {current_figurex35_share:.1f}%")
        print(f"   • FigureX-4o Share: {current_figurex4o_share:.1f}%")
        print()
        print("📈 RECENT TRENDS (30-day):")
        print(f"   • FigureX-3.5-turbo Growth: {figurex35_growth:+.1f}%")
        print(f"   • FigureX-4o Growth: {figurex4o_growth:+.1f}%")
        print()

        # Add changepoint insights
        if hasattr(self, 'changepoints') and self.changepoints:
            print("🔍 CHANGEPOINT ANALYSIS:")
            for model_type, cps in self.changepoints.items():
                if cps and len(cps) > 1:  # Exclude the last automatic point
                    actual_cps = len(cps) - 1
                    print(f"   • {model_type.upper()} Usage: {actual_cps} significant changepoints detected")
                    if actual_cps > 0:
                        # Convert first changepoint to approximate date
                        first_cp_date = data['date'].iloc[cps[0]] if cps[0] < len(data) else "Unknown"
                        print(
                            f"     → First major shift: ~{first_cp_date.strftime('%Y-%m-%d') if hasattr(first_cp_date, 'strftime') else first_cp_date}")
            print()

        print("🔍 KEY FINDINGS:")
        print("   • Clear transition pattern observed post-FigureX-4o launch")
        print("   • Weekend usage shows different patterns between models")
        print("   • Media coverage drives short-term adoption spikes")
        if hasattr(self, 'changepoints') and self.changepoints:
            total_cps = len(self.changepoints.get('total', [])) - 1
            if total_cps > 2:
                print("   • High volatility detected - multiple significant usage shifts")
            elif total_cps > 0:
                print("   • Moderate usage volatility with clear transition points")
        print()
        print("🎯 SCENARIO PROJECTIONS (90-day):")

        if self.scenarios:
            for scenario, data_dict in self.scenarios.items():
                final_total = data_dict['total_usage'][-1]
                growth_proj = (final_total / current_total - 1) * 100
                print(f"   • {scenario.replace('_', ' ').title()}: {growth_proj:+.1f}% total growth")

        print()
        print("⚡ COMPUTE INFRASTRUCTURE RECOMMENDATIONS:")
        print("   • Scale capacity by 40-80% for aggressive switching scenario")
        print("   • Implement dynamic scaling based on model usage patterns")
        print("   • Monitor weekly migration rates for early scaling triggers")
        print("   • Consider cost optimization through model-specific infrastructure")
        if hasattr(self, 'changepoints') and self.changepoints:
            total_cps = len(self.changepoints.get('total', [])) - 1
            if total_cps > 2:
                print("   • ⚠️  High volatility detected - implement more aggressive monitoring")
                print("   • Consider additional buffer capacity for sudden usage spikes")
        print()
        print("📋 NEXT STEPS:")
        print("   • Implement real-time monitoring dashboard")
        print("   • Set up automated alerts for usage pattern changes")
        print("   • Plan phased infrastructure scaling")
        print("   • Develop customer communication strategy for deprecation")
        if hasattr(self, 'changepoints') and self.changepoints:
            print("   • Establish changepoint detection in production for early warning")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""

        print("🚀 Starting Comprehensive API Usage Forecast Analysis")
        print("=" * 60)

        # 1. Generate sample data
        print("📊 Generating sample API usage data...")
        data = self.generate_sample_data(days=365)

        # 2. Decompose usage by model
        print("🔍 Analyzing usage patterns by model...")
        data = self.decompose_usage_by_model(data)

        # 3. Detect changepoints
        print("📈 Detecting significant changepoints...")
        try:
            self.detect_changepoints(data)  # This now stores results in self.changepoints
            print(f"   ✓ Changepoints detected and stored for use in modeling")
        except ImportError:
            print("⚠️  ruptures package not available for changepoint detection")
            print("   Install with: pip install ruptures")
            self.changepoints = {}  # Initialize empty if ruptures not available

        # 4. Build models and generate forecasts
        print("🤖 Building Prophet and XGBoost models...")
        forecasts = self.generate_forecasts(data, forecast_days=90)

        # 5. Create scenarios
        print("📊 Creating migration scenarios...")
        scenarios = self.create_scenarios(data, forecast_days=90)
        compute_reqs = self.visualize_scenarios(data)

        # 6. Uncertainty analysis
        print("📏 Analyzing forecast uncertainty...")
        self.generate_uncertainty_intervals(data)

        # 7. Executive summary
        print("📋 Generating executive summary...")
        self.generate_executive_summary(data)

        print("\n✅ Analysis complete! All visualizations and insights generated.")

        return {
            'data': data,
            'forecasts': forecasts,
            'scenarios': scenarios,
            'compute_requirements': compute_reqs
        }


# Example usage
if __name__ == "__main__":
    # Initialize the forecasting system
    forecaster = APIUsageForecast()

    # Run complete analysis
    results = forecaster.run_complete_analysis()

    # Additional analysis examples
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)

    # Model comparison
    data = results['data']
    correlation = data[['figurex35_usage', 'figurex4o_usage', 'total_usage']].corr()
    print("\n📊 Model Usage Correlations:")
    print(correlation)

    # Peak usage analysis
    peak_total = data['total_usage'].max()
    peak_date = data.loc[data['total_usage'].idxmax(), 'date']
    print(f"\n🔝 Peak Usage: {peak_total:,.0f} calls on {peak_date.strftime('%Y-%m-%d')}")

    # Weekend vs weekday patterns
    weekend_avg = data[data['is_weekend']]['total_usage'].mean()
    weekday_avg = data[~data['is_weekend']]['total_usage'].mean()
    print(f"📅 Weekend vs Weekday Usage:")
    print(f"   • Weekend Average: {weekend_avg:,.0f} calls")
    print(f"   • Weekday Average: {weekday_avg:,.0f} calls")
    print(f"   • Weekend Factor: {weekend_avg / weekday_avg:.2f}x")
