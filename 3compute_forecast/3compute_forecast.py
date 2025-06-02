# 🧠 Compute Demand Forecasting
# FigureX.ai is scaling infrastructure based on projected compute usage from FigureX API traffic.
# You're asked to forecast future compute demand.
# The catch: traffic is highly bursty and new features often cause abrupt usage spikes.
# What’s your approach?

# Combine short-term (high-variance) + long-term (trend) models
# Use ensemble models: ARIMA for baseline + anomaly detection for bursts
# Add regressors for major feature rollouts, partner deals, holidays
# Consider autoregressive models + external covariates (e.g., promo events)
# Use model monitoring: track drift, set alerts for major deviation


import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


class ComputeDemandForecaster:
    """
    Forecasts compute demand for FigureX.ai API infrastructure with special handling for:
    1. Bursty traffic patterns
    2. Feature launch spikes
    3. Capacity planning with safety margins
    4. Real-time anomaly detection
    """

    def __init__(self):
        self.models = {}
        self.spike_detector = None
        self.baseline_model = None
        self.spike_model = None
        self.capacity_buffer = 1.3  # 30% safety margin
        self.feature_launches = []
        self.historical_spikes = []

    def generate_synthetic_compute_data(self, start_date='2023-01-01', end_date='2024-12-31'):
        """Generate realistic compute usage data with burstiness and spikes"""

        dates = pd.date_range(start=start_date, end=end_date, freq='H')  # Hourly data
        n_hours = len(dates)

        # Base components
        trend = np.linspace(100, 500, n_hours)  # Growing from 100 to 500 TFLOPS

        # Multiple seasonality patterns
        daily_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_hours) / 24)  # Daily cycle
        weekly_pattern = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 7))  # Weekly cycle
        monthly_pattern = 1 + 0.1 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 30))  # Monthly cycle

        # Business hours effect (higher during work hours)
        business_hours = np.array([
            1.5 if (dates[i].hour >= 9 and dates[i].hour <= 17 and dates[i].weekday() < 5) else 1.0
            for i in range(n_hours)
        ])

        # Base usage combining all patterns
        base_usage = trend * daily_pattern * weekly_pattern * monthly_pattern * business_hours

        # Add random noise with heavy tail for burstiness
        noise = np.random.lognormal(0, 0.3, n_hours)

        # Feature launches and their impacts
        feature_launches = [
            {'date': '2023-03-14', 'name': 'FX-4 Launch', 'spike_multiplier': 5.0, 'decay_days': 7},
            {'date': '2023-05-12', 'name': 'Plugins Release', 'spike_multiplier': 3.0, 'decay_days': 5},
            {'date': '2023-07-20', 'name': 'Code Interpreter', 'spike_multiplier': 2.5, 'decay_days': 4},
            {'date': '2023-09-25', 'name': 'DALL-E 3 Integration', 'spike_multiplier': 4.0, 'decay_days': 6},
            {'date': '2023-11-06', 'name': 'FX-4 Turbo', 'spike_multiplier': 6.0, 'decay_days': 10},
            {'date': '2024-01-10', 'name': 'FXs Store', 'spike_multiplier': 3.5, 'decay_days': 5},
            {'date': '2024-03-01', 'name': 'Real-time Voice', 'spike_multiplier': 4.5, 'decay_days': 7},
            {'date': '2024-05-13', 'name': 'FX-4o Release', 'spike_multiplier': 7.0, 'decay_days': 14},
        ]

        # Random viral events (unpredictable spikes)
        viral_events = []
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        date_range_days = (end_date_dt - start_date_dt).days

        for i in range(15):  # 15 random viral events
            random_days = np.random.randint(0, date_range_days)
            random_date = start_date_dt + timedelta(days=random_days)
            viral_events.append({
                'date': random_date.strftime('%Y-%m-%d'),
                'spike_multiplier': np.random.uniform(2.0, 4.0),
                'decay_days': np.random.randint(1, 4)
            })

        # Apply feature launch spikes
        compute_usage = base_usage * noise

        for event in feature_launches + viral_events:
            event_date = pd.to_datetime(event['date'])
            if pd.to_datetime(start_date) <= event_date <= pd.to_datetime(end_date):
                event_hour = (event_date - pd.to_datetime(start_date)).days * 24

                # Create spike with exponential decay
                for h in range(event_hour, min(event_hour + event['decay_days'] * 24, n_hours)):
                    hours_since_launch = h - event_hour
                    decay_factor = np.exp(-hours_since_launch / (event['decay_days'] * 24 / 3))
                    spike_effect = 1 + (event['spike_multiplier'] - 1) * decay_factor
                    compute_usage[h] *= spike_effect

        # Add random bursts (API abuse, DDoS-like patterns)
        n_bursts = int(n_hours * 0.01)  # 1% of hours have bursts
        burst_hours = np.random.choice(n_hours, n_bursts, replace=False)
        for hour in burst_hours:
            burst_duration = np.random.randint(1, 6)  # 1-6 hour bursts
            burst_intensity = np.random.uniform(2.0, 5.0)
            for h in range(hour, min(hour + burst_duration, n_hours)):
                compute_usage[h] *= burst_intensity

        # Create dataframe with additional features
        df = pd.DataFrame({
            'timestamp': dates,
            'compute_tflops': compute_usage,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'is_weekend': dates.weekday >= 5,
            'is_business_hours': (dates.hour >= 9) & (dates.hour <= 17) & (dates.weekday < 5),
            'month': dates.month,
            'week_of_year': dates.isocalendar().week,
        })

        # Add API-specific metrics
        df['request_count'] = df['compute_tflops'] * np.random.uniform(1000, 1500, n_hours)
        df['avg_tokens_per_request'] = np.random.normal(500, 100, n_hours)
        df['p95_latency_ms'] = 50 + df['compute_tflops'] * 0.1 + np.random.normal(0, 10, n_hours)
        df['queue_depth'] = np.maximum(0, df['compute_tflops'] - 400) * 10 + np.random.normal(0, 50, n_hours)

        # Store feature launches for later use
        self.feature_launches = pd.DataFrame(feature_launches)
        self.feature_launches['date'] = pd.to_datetime(self.feature_launches['date'])

        return df

    def detect_anomalies_and_spikes(self, df):
        """Detect anomalies and classify spike types"""

        # Use multiple methods for robustness

        # 1. Statistical method (z-score on rolling window)
        window = 24 * 7  # 1 week window
        rolling_mean = df['compute_tflops'].rolling(window, center=True).mean()
        rolling_std = df['compute_tflops'].rolling(window, center=True).std()
        z_scores = (df['compute_tflops'] - rolling_mean) / rolling_std

        # 2. Isolation Forest for multivariate anomaly detection
        features = ['compute_tflops', 'request_count', 'queue_depth', 'p95_latency_ms']
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = iso_forest.fit_predict(df[features].fillna(0))

        # 3. Percent change detection
        pct_change = df['compute_tflops'].pct_change()

        # Classify spikes
        df['is_anomaly'] = (z_scores.abs() > 3) | (anomaly_labels == -1)
        df['spike_type'] = 'normal'

        # Feature launch spikes (check proximity to known launches)
        for _, launch in self.feature_launches.iterrows():
            mask = (df['timestamp'] >= launch['date']) & \
                   (df['timestamp'] <= launch['date'] + timedelta(days=launch['decay_days']))
            df.loc[mask & df['is_anomaly'], 'spike_type'] = 'feature_launch'

        # Sudden spikes (viral/abuse)
        sudden_spike_mask = (pct_change > 0.5) & df['is_anomaly']
        df.loc[sudden_spike_mask & (df['spike_type'] == 'normal'), 'spike_type'] = 'sudden_spike'

        # Gradual increase (organic growth spurts)
        gradual_mask = (z_scores > 2) & (z_scores < 3) & (pct_change < 0.3)
        df.loc[gradual_mask, 'spike_type'] = 'gradual_increase'

        return df

    def create_advanced_features(self, df):
        """Create features for forecasting including spike indicators"""

        df = df.copy()

        # Reset index to ensure we have integer indices for calculations
        df = df.reset_index(drop=True)

        # Lag features at multiple scales
        for lag in [1, 6, 12, 24, 48, 168]:  # 1h, 6h, 12h, 1d, 2d, 1w
            df[f'compute_lag_{lag}h'] = df['compute_tflops'].shift(lag)
            df[f'request_lag_{lag}h'] = df['request_count'].shift(lag)

        # Rolling statistics
        for window in [6, 24, 168]:  # 6h, 1d, 1w
            df[f'compute_rolling_mean_{window}h'] = df['compute_tflops'].rolling(window).mean()
            df[f'compute_rolling_std_{window}h'] = df['compute_tflops'].rolling(window).std()
            df[f'compute_rolling_max_{window}h'] = df['compute_tflops'].rolling(window).max()
            df[f'compute_rolling_p95_{window}h'] = df['compute_tflops'].rolling(window).quantile(0.95)

        # Growth features
        df['compute_growth_1h'] = df['compute_tflops'].pct_change()
        df['compute_growth_24h'] = df['compute_tflops'].pct_change(24)
        df['compute_growth_168h'] = df['compute_tflops'].pct_change(168)

        # Spike history features - FIXED VERSION
        df['hours_since_last_spike'] = 0.0

        # Get indices of spikes (use integer positions, not index values)
        spike_mask = df['is_anomaly'].fillna(False)
        spike_positions = df.index[spike_mask].tolist()

        for i in range(len(df)):
            if i in spike_positions:
                df.loc[i, 'hours_since_last_spike'] = 0.0
            else:
                # Find the most recent spike before current position
                prev_spikes = [s for s in spike_positions if s < i]
                if len(prev_spikes) > 0:
                    df.loc[i, 'hours_since_last_spike'] = float(i - prev_spikes[-1])
                else:
                    df.loc[i, 'hours_since_last_spike'] = float(i)  # Hours since start if no previous spike

        # Time-based features with cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Capacity pressure indicators
        df['capacity_utilization'] = df['compute_tflops'] / df['compute_tflops'].rolling(168).max().fillna(
            df['compute_tflops'])
        df['queue_pressure'] = df['queue_depth'] / df['queue_depth'].rolling(168).mean().fillna(df['queue_depth'])

        # Ensure all numeric columns are float type (exclude non-numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        return df

    def fit_hybrid_models(self, df):
        """Fit ensemble of models for different scenarios"""

        # Prepare data
        df = self.detect_anomalies_and_spikes(df)
        df = self.create_advanced_features(df)

        # Split normal vs spike data
        normal_data = df[df['spike_type'] == 'normal'].copy()

        # 1. Prophet for baseline (normal traffic)
        prophet_df = normal_data[['timestamp', 'compute_tflops']].rename(
            columns={'timestamp': 'ds', 'compute_tflops': 'y'}
        )

        self.baseline_model = Prophet(
            growth='linear',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )

        # Add custom seasonalities
        self.baseline_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.baseline_model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

        self.baseline_model.fit(prophet_df)

        # 2. XGBoost for spike prediction
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'compute_tflops', 'spike_type', 'is_anomaly'
        ]]

        # Prepare training data
        X = df[feature_cols].fillna(0)
        y_spike_probability = (df['spike_type'] != 'normal').astype(int)
        y_spike_magnitude = df['compute_tflops'] / df['compute_rolling_mean_168h'].fillna(df['compute_tflops'])

        # Spike probability model
        self.spike_probability_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.spike_probability_model.fit(X, y_spike_probability)

        # Spike magnitude model (only on spike data)
        spike_indices = df[df['spike_type'] != 'normal'].index
        if len(spike_indices) > 100:
            X_spike = X.loc[spike_indices]
            y_magnitude = y_spike_magnitude.loc[spike_indices]

            self.spike_magnitude_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            self.spike_magnitude_model.fit(X_spike, y_magnitude)

        # 3. Quantile regression for capacity planning
        self.quantile_models = {}
        for quantile in [0.5, 0.75, 0.9, 0.95, 0.99]:
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=quantile,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(X, df['compute_tflops'])
            self.quantile_models[quantile] = model

        self.feature_cols = feature_cols
        return self

    def forecast_with_scenarios(self, df, forecast_hours=24 * 7, include_planned_launches=None):
        """Generate forecasts with multiple scenarios and capacity recommendations"""

        # Prepare base forecast
        last_timestamp = df['timestamp'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=forecast_hours,
            freq='H'
        )

        # 1. Baseline forecast (Prophet)
        future_df = pd.DataFrame({'ds': future_timestamps})
        baseline_forecast = self.baseline_model.predict(future_df)

        # 2. Create future features for advanced models
        future_features = pd.DataFrame({
            'timestamp': future_timestamps,
            'hour': future_timestamps.hour,
            'day_of_week': future_timestamps.dayofweek,
            'is_weekend': future_timestamps.weekday >= 5,
            'is_business_hours': (future_timestamps.hour >= 9) &
                                 (future_timestamps.hour <= 17) &
                                 (future_timestamps.weekday < 5),
            'month': future_timestamps.month,
            'week_of_year': future_timestamps.isocalendar().week,
        })

        # Add cyclical features
        future_features['hour_sin'] = np.sin(2 * np.pi * future_features['hour'] / 24)
        future_features['hour_cos'] = np.cos(2 * np.pi * future_features['hour'] / 24)
        future_features['dow_sin'] = np.sin(2 * np.pi * future_features['day_of_week'] / 7)
        future_features['dow_cos'] = np.cos(2 * np.pi * future_features['day_of_week'] / 7)
        future_features['month_sin'] = np.sin(2 * np.pi * future_features['month'] / 12)
        future_features['month_cos'] = np.cos(2 * np.pi * future_features['month'] / 12)

        # Use last known values for lag features
        last_values = df.iloc[-1]
        for col in self.feature_cols:
            if col not in future_features.columns:
                if 'lag' in col or 'rolling' in col:
                    future_features[col] = last_values[col] if col in last_values else 0
                else:
                    future_features[col] = 0

        # 3. Predict spike probability
        X_future = future_features[self.feature_cols].fillna(0)
        spike_probabilities = self.spike_probability_model.predict_proba(X_future)[:, 1]

        # 4. Generate quantile forecasts
        quantile_forecasts = {}
        for quantile, model in self.quantile_models.items():
            quantile_forecasts[f'p{int(quantile * 100)}'] = model.predict(X_future)

        # 5. Combine into scenarios
        scenarios = pd.DataFrame({
            'timestamp': future_timestamps,
            'baseline': baseline_forecast['yhat'].values,
            'baseline_lower': baseline_forecast['yhat_lower'].values,
            'baseline_upper': baseline_forecast['yhat_upper'].values,
            'spike_probability': spike_probabilities,
            'expected_value': baseline_forecast['yhat'].values * (1 - spike_probabilities) +
                              baseline_forecast['yhat'].values * 2.5 * spike_probabilities,  # Expected with spikes
            **quantile_forecasts
        })

        # 6. Add planned feature launches if provided
        if include_planned_launches:
            for launch in include_planned_launches:
                launch_date = pd.to_datetime(launch['date'])
                if launch_date in scenarios['timestamp'].values:
                    idx = scenarios[scenarios['timestamp'] == launch_date].index[0]
                    # Simulate launch spike
                    for i in range(idx, min(idx + launch['expected_duration_hours'], len(scenarios))):
                        decay = np.exp(-(i - idx) / (launch['expected_duration_hours'] / 3))
                        spike_factor = 1 + (launch['expected_multiplier'] - 1) * decay
                        scenarios.loc[i, 'expected_value'] *= spike_factor
                        scenarios.loc[i, 'p95'] *= spike_factor
                        scenarios.loc[i, 'p99'] *= spike_factor

        # 7. Capacity recommendations
        scenarios['recommended_capacity'] = scenarios['p95'] * self.capacity_buffer
        scenarios['burst_capacity'] = scenarios['p99'] * 1.5  # Extra buffer for extreme events

        return scenarios

    def calculate_infrastructure_requirements(self, scenarios, gpu_tflops=312):
        """Calculate actual infrastructure needs based on forecasts"""

        # Assume each H100 GPU provides ~312 TFLOPS
        requirements = pd.DataFrame({
            'timestamp': scenarios['timestamp'],
            'baseline_gpus': np.ceil(scenarios['baseline'] / gpu_tflops),
            'expected_gpus': np.ceil(scenarios['expected_value'] / gpu_tflops),
            'recommended_gpus': np.ceil(scenarios['recommended_capacity'] / gpu_tflops),
            'burst_ready_gpus': np.ceil(scenarios['burst_capacity'] / gpu_tflops),
        })

        # Add cost estimates (simplified)
        gpu_hour_cost = 3.5  # Rough estimate for H100
        requirements['baseline_cost_per_hour'] = requirements['baseline_gpus'] * gpu_hour_cost
        requirements['recommended_cost_per_hour'] = requirements['recommended_gpus'] * gpu_hour_cost
        requirements['burst_cost_per_hour'] = requirements['burst_ready_gpus'] * gpu_hour_cost

        # Utilization metrics
        requirements['expected_utilization'] = scenarios['expected_value'] / scenarios['recommended_capacity']
        requirements['burst_utilization'] = scenarios['expected_value'] / scenarios['burst_capacity']

        return requirements

    def plot_forecast_analysis(self, df, scenarios, requirements):
        """Create comprehensive visualization of forecasts and capacity planning"""

        fig, axes = plt.subplots(4, 2, figsize=(20, 20))

        # 1. Historical compute usage with anomalies
        ax1 = axes[0, 0]
        ax1.plot(df['timestamp'], df['compute_tflops'], label='Actual Usage', alpha=0.7)

        # Highlight different spike types
        spike_types = df['spike_type'].unique()
        colors = {'feature_launch': 'red', 'sudden_spike': 'orange', 'gradual_increase': 'yellow'}
        for spike_type in spike_types:
            if spike_type != 'normal':
                spike_data = df[df['spike_type'] == spike_type]
                ax1.scatter(spike_data['timestamp'], spike_data['compute_tflops'],
                            c=colors.get(spike_type, 'gray'), label=spike_type, alpha=0.6, s=20)

        ax1.set_title('Historical Compute Usage with Anomalies')
        ax1.set_ylabel('Compute (TFLOPS)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Forecast scenarios
        ax2 = axes[0, 1]
        ax2.plot(scenarios['timestamp'], scenarios['baseline'], label='Baseline', color='blue')
        ax2.fill_between(scenarios['timestamp'], scenarios['baseline_lower'],
                         scenarios['baseline_upper'], alpha=0.2, color='blue')
        ax2.plot(scenarios['timestamp'], scenarios['expected_value'],
                 label='Expected (with spikes)', color='red', linewidth=2)
        ax2.plot(scenarios['timestamp'], scenarios['p95'],
                 label='95th Percentile', color='orange', linestyle='--')
        ax2.plot(scenarios['timestamp'], scenarios['p99'],
                 label='99th Percentile', color='darkred', linestyle=':')

        ax2.set_title('Compute Demand Forecast Scenarios')
        ax2.set_ylabel('Compute (TFLOPS)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Spike probability
        ax3 = axes[1, 0]
        ax3.plot(scenarios['timestamp'], scenarios['spike_probability'] * 100)
        ax3.fill_between(scenarios['timestamp'], 0, scenarios['spike_probability'] * 100, alpha=0.3)
        ax3.set_title('Predicted Spike Probability')
        ax3.set_ylabel('Probability (%)')
        ax3.grid(True, alpha=0.3)

        # 4. Infrastructure requirements
        ax4 = axes[1, 1]
        ax4.plot(requirements['timestamp'], requirements['baseline_gpus'],
                 label='Baseline', color='blue')
        ax4.plot(requirements['timestamp'], requirements['expected_gpus'],
                 label='Expected', color='green')
        ax4.plot(requirements['timestamp'], requirements['recommended_gpus'],
                 label='Recommended', color='orange', linewidth=2)
        ax4.plot(requirements['timestamp'], requirements['burst_ready_gpus'],
                 label='Burst Ready', color='red', linestyle='--')

        ax4.set_title('GPU Requirements')
        ax4.set_ylabel('Number of GPUs')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Cost analysis
        ax5 = axes[2, 0]
        daily_costs = requirements.groupby(requirements['timestamp'].dt.date).agg({
            'baseline_cost_per_hour': 'sum',
            'recommended_cost_per_hour': 'sum',
            'burst_cost_per_hour': 'sum'
        })

        x = range(len(daily_costs))
        width = 0.25
        ax5.bar([i - width for i in x], daily_costs['baseline_cost_per_hour'],
                width, label='Baseline', color='blue')
        ax5.bar(x, daily_costs['recommended_cost_per_hour'],
                width, label='Recommended', color='orange')
        ax5.bar([i + width for i in x], daily_costs['burst_cost_per_hour'],
                width, label='Burst Ready', color='red')

        ax5.set_title('Daily Infrastructure Costs')
        ax5.set_ylabel('Cost ($)')
        ax5.set_xticklabels(daily_costs.index, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Utilization efficiency
        ax6 = axes[2, 1]
        ax6.plot(requirements['timestamp'], requirements['expected_utilization'] * 100,
                 label='Recommended Capacity', color='green')
        ax6.plot(requirements['timestamp'], requirements['burst_utilization'] * 100,
                 label='Burst Capacity', color='orange')
        ax6.axhline(y=70, color='red', linestyle='--', label='Target Utilization')

        ax6.set_title('Infrastructure Utilization')
        ax6.set_ylabel('Utilization (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Capacity planning summary
        ax7 = axes[3, 0]
        summary_data = {
            'Metric': ['Avg Daily Compute', 'Peak Compute', 'Avg GPUs Needed',
                       'Peak GPUs Needed', 'Monthly Cost'],
            'Baseline': [
                f"{scenarios['baseline'].mean():.0f} TFLOPS",
                f"{scenarios['baseline'].max():.0f} TFLOPS",
                f"{requirements['baseline_gpus'].mean():.0f}",
                f"{requirements['baseline_gpus'].max():.0f}",
                f"${requirements['baseline_cost_per_hour'].sum() * 24 * 30:.0f}"
            ],
            'Recommended': [
                f"{scenarios['recommended_capacity'].mean():.0f} TFLOPS",
                f"{scenarios['recommended_capacity'].max():.0f} TFLOPS",
                f"{requirements['recommended_gpus'].mean():.0f}",
                f"{requirements['recommended_gpus'].max():.0f}",
                f"${requirements['recommended_cost_per_hour'].sum() * 24 * 30:.0f}"
            ]
        }

        ax7.axis('tight')
        ax7.axis('off')
        table = ax7.table(cellText=[summary_data['Baseline'], summary_data['Recommended']],
                          rowLabels=['Baseline', 'Recommended'],
                          colLabels=summary_data['Metric'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax7.set_title('Capacity Planning Summary', pad=20)

        # 8. Risk assessment
        ax8 = axes[3, 1]
        risk_levels = {
            'Low Risk\n(Normal Operations)': (scenarios['spike_probability'] < 0.1).sum() / len(scenarios) * 100,
            'Medium Risk\n(Potential Spikes)': ((scenarios['spike_probability'] >= 0.1) &
                                                (scenarios['spike_probability'] < 0.3)).sum() / len(scenarios) * 100,
            'High Risk\n(Likely Spikes)': (scenarios['spike_probability'] >= 0.3).sum() / len(scenarios) * 100
        }

        colors_risk = ['green', 'orange', 'red']
        ax8.pie(risk_levels.values(), labels=risk_levels.keys(), colors=colors_risk,
                autopct='%1.1f%%', startangle=90)
        ax8.set_title('Risk Distribution for Forecast Period')

        plt.suptitle('FigureX.ai Compute Demand Forecast & Capacity Planning', fontsize=20)
        plt.tight_layout()
        plt.savefig('3compute_forecast/compute_demand_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_strategic_recommendations(self, df, scenarios, requirements):
        """Generate actionable recommendations for infrastructure scaling"""

        # Analyze patterns
        avg_utilization = requirements['expected_utilization'].mean()
        peak_gpus = requirements['recommended_gpus'].max()
        avg_gpus = requirements['recommended_gpus'].mean()
        spike_frequency = (scenarios['spike_probability'] > 0.2).sum() / len(scenarios)

        # Cost analysis
        monthly_baseline_cost = requirements['baseline_cost_per_hour'].sum() * 24 * 30
        monthly_recommended_cost = requirements['recommended_cost_per_hour'].sum() * 24 * 30
        cost_increase = (monthly_recommended_cost - monthly_baseline_cost) / monthly_baseline_cost * 100

        recommendations = f"""
STRATEGIC INFRASTRUCTURE RECOMMENDATIONS
========================================

📊 CAPACITY PLANNING:
• Current Baseline: {int(requirements['baseline_gpus'].mean())} GPUs average
• Recommended Capacity: {int(avg_gpus)} GPUs average ({int(peak_gpus)} peak)
• Safety Buffer: 30% above P95 (protects against 95% of spikes)
• Burst Capacity: {int(requirements['burst_ready_gpus'].max())} GPUs (for extreme events)

💰 COST ANALYSIS:
• Monthly Baseline Cost: ${monthly_baseline_cost:,.0f}
• Monthly Recommended Cost: ${monthly_recommended_cost:,.0f}
• Cost Increase: {cost_increase:.1f}% for spike protection
• Cost per prevented outage: ${(monthly_recommended_cost - monthly_baseline_cost) / (spike_frequency * 30):,.0f}

🎯 SCALING STRATEGY:
1. IMMEDIATE ACTIONS (0-2 weeks):
   - Provision {int(avg_gpus - requirements['baseline_gpus'].mean())} additional GPUs
   - Set up auto-scaling triggers at 70% utilization
   - Configure burst capacity contracts with cloud providers

2. SHORT-TERM (1-3 months):
   - Implement predictive scaling based on spike probability model
   - Deploy A/B testing for graceful degradation during spikes
   - Establish dedicated capacity for feature launches

3. LONG-TERM (3-6 months):
   - Build multi-region failover for geographic load distribution
   - Develop adaptive request routing based on real-time capacity
   - Create tiered service levels (priority queues for enterprise)

⚠️ RISK MITIGATION:
• High Risk Hours: {(scenarios['spike_probability'] > 0.3).sum()} hours in next week
• Spike Frequency: {spike_frequency * 100:.1f}% of time
• Average Utilization: {avg_utilization * 100:.1f}% (target: 70%)

🔄 OPTIMIZATION OPPORTUNITIES:
1. Request Batching: Could reduce compute by 10-15% during normal operations
2. Model Quantization: 20-30% efficiency gain with minimal quality loss
3. Caching Strategy: Reduce repeated computations by 25%

📈 MONITORING METRICS:
• P95 latency threshold: <500ms
• Queue depth alarm: >1000 requests
• Utilization warning: >85%
• Spike detection: 2x rolling average

🚨 EMERGENCY PROTOCOLS:
1. Spike Detection → Auto-scale within 2 minutes
2. Capacity >90% → Activate burst reserves
3. Queue >5000 → Enable request throttling
4. Feature Launch → Pre-provision 3x normal capacity
"""

        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = ComputeDemandForecaster()

    # Generate synthetic compute data
    print("Generating synthetic compute demand data...")
    compute_data = forecaster.generate_synthetic_compute_data(
        start_date='2023-01-01',
        end_date='2024-12-31'
    )

    print(f"Generated {len(compute_data)} hours of compute data")
    print(f"Average compute: {compute_data['compute_tflops'].mean():.2f} TFLOPS")
    print(f"Peak compute: {compute_data['compute_tflops'].max():.2f} TFLOPS")

    # Detect anomalies
    print("\nDetecting anomalies and spikes...")
    compute_data = forecaster.detect_anomalies_and_spikes(compute_data)
    spike_summary = compute_data['spike_type'].value_counts()
    print("Spike distribution:")
    for spike_type, count in spike_summary.items():
        print(f"  - {spike_type}: {count} hours ({count / len(compute_data) * 100:.1f}%)")

    # Fit models
    print("\nFitting forecasting models...")
    forecaster.fit_hybrid_models(compute_data)

    # Generate forecasts with planned launches
    print("\nGenerating forecasts for next 7 days...")
    planned_launches = [
        {
            'date': '2025-01-15 09:00:00',
            'name': 'FX-5 Preview',
            'expected_multiplier': 5.0,
            'expected_duration_hours': 72
        }
    ]

    scenarios = forecaster.forecast_with_scenarios(
        compute_data,
        forecast_hours=24 * 7,
        include_planned_launches=planned_launches
    )

    # Calculate infrastructure requirements
    print("\nCalculating infrastructure requirements...")
    requirements = forecaster.calculate_infrastructure_requirements(scenarios)

    # Generate visualizations
    print("\nCreating visualizations...")
    forecaster.plot_forecast_analysis(compute_data, scenarios, requirements)

    # Generate recommendations
    print("\nGenerating strategic recommendations...")
    recommendations = forecaster.generate_strategic_recommendations(
        compute_data, scenarios, requirements
    )
    print(recommendations)

    # Additional analysis outputs
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    print(f"""
📊 Compute Demand Forecast Summary:
• Baseline Growth: {(scenarios['baseline'].iloc[-1] / scenarios['baseline'].iloc[0] - 1) * 100:.1f}% over next week
• Expected Spikes: {(scenarios['spike_probability'] > 0.2).sum()} high-risk hours
• Peak Demand: {scenarios['expected_value'].max():.0f} TFLOPS
• Infrastructure Recommendation: {int(requirements['recommended_gpus'].mean())} GPUs average
• Monthly Cost Impact: ${(requirements['recommended_cost_per_hour'].sum() * 24 * 30):,.0f}

🎯 Key Insights:
1. Feature launches create 5-7x traffic spikes lasting 3-14 days
2. Viral events are unpredictable but typically 2-4x normal traffic
3. Business hours show 50% higher usage than off-hours
4. Weekend traffic is 30% lower on average

⚡ Critical Actions:
1. Maintain 30% capacity buffer above P95 predictions
2. Pre-scale 3x for announced feature launches
3. Enable auto-scaling at 70% utilization threshold
4. Keep burst capacity contracts for 99th percentile events
""")

    # Export results
    print("\nExporting results...")
    scenarios.to_csv('3compute_forecast/compute_demand_scenarios.csv', index=False)
    requirements.to_csv('3compute_forecast/infrastructure_requirements.csv', index=False)
    print("Results exported to CSV files")

    # Performance metrics
    feature_importance = forecaster.spike_probability_model.feature_importances_
    top_features = sorted(
        zip(forecaster.feature_cols, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("\nTop 10 Features for Spike Prediction:")
    for feature, importance in top_features:
        print(f"  - {feature}: {importance:.4f}")
