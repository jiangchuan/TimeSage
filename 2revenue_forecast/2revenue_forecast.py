# 💰 Revenue Forecast for Strategic Planning
# Finance wants a quarterly revenue forecast to guide hiring and infrastructure spend.
# You’re given daily revenue data segmented by product and geography.
# How do you model and communicate your forecast?

# Aggregation: Daily → weekly or monthly for stability
# Hierarchical forecasting: by product/geo → roll up
# Consider ETS, Prophet, or LightGBM with time-aware features
# Quantify risk using prediction intervals or scenario modeling
# Explain assumptions clearly: seasonality, retention, pricing
# Discuss how this forecast will feed into planning decisions

import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class RevenueForecaster:
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.scalers = {}
        self.feature_importance = {}
        self.cross_effects = {
            'plus_to_api': -0.05,
            'enterprise_to_api': 0.15,
            'api_to_plus': 0.03,
            'enterprise_to_plus': -0.02
        }

    def generate_synthetic_revenue_data(self, start_date='2022-11-01', end_date='2024-12-31'):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)

        products = ['figurex_plus', 'api', 'enterprise']
        geographies = ['north_america', 'europe', 'apac', 'other']

        base_params = {
            # FigureX.ai Plus (Subscription model)
            ('figurex_plus', 'north_america'): {'initial': 50000, 'growth': 0.012, 'seasonality': 0.05},
            ('figurex_plus', 'europe'): {'initial': 25000, 'growth': 0.015, 'seasonality': 0.08},
            ('figurex_plus', 'apac'): {'initial': 15000, 'growth': 0.025, 'seasonality': 0.10},
            ('figurex_plus', 'other'): {'initial': 10000, 'growth': 0.020, 'seasonality': 0.12},

            # API (Usage-based model)
            ('api', 'north_america'): {'initial': 75000, 'growth': 0.018, 'seasonality': 0.03},
            ('api', 'europe'): {'initial': 35000, 'growth': 0.022, 'seasonality': 0.04},
            ('api', 'apac'): {'initial': 25000, 'growth': 0.030, 'seasonality': 0.05},
            ('api', 'other'): {'initial': 15000, 'growth': 0.025, 'seasonality': 0.06},

            # Enterprise (Contract-based model)
            ('enterprise', 'north_america'): {'initial': 100000, 'growth': 0.008, 'seasonality': 0.12},
            ('enterprise', 'europe'): {'initial': 40000, 'growth': 0.010, 'seasonality': 0.15},
            ('enterprise', 'apac'): {'initial': 20000, 'growth': 0.020, 'seasonality': 0.18},
            ('enterprise', 'other'): {'initial': 10000, 'growth': 0.015, 'seasonality': 0.20},
        }

        # Generate product launch events and their impacts
        events = [
            # Major product launches
            {'date': '2022-12-01', 'type': 'figurex_launch', 'products': ['figurex_plus'], 'impact': 2.5,
             'duration': 30},
            {'date': '2023-03-14', 'type': 'fx4_launch', 'products': ['figurex_plus', 'api'], 'impact': 2.0,
             'duration': 45},
            {'date': '2023-11-06', 'type': 'devday_2023', 'products': ['api', 'enterprise'], 'impact': 1.8,
             'duration': 30},
            {'date': '2024-05-13', 'type': 'fx4o_launch', 'products': ['figurex_plus', 'api'], 'impact': 1.6,
             'duration': 21},

            # Seasonal events
            {'date': '2023-11-24', 'type': 'black_friday', 'products': ['figurex_plus'], 'impact': 1.3, 'duration': 7},
            {'date': '2023-12-31', 'type': 'year_end_enterprise', 'products': ['enterprise'], 'impact': 1.6,
             'duration': 14},
            {'date': '2024-11-29', 'type': 'black_friday', 'products': ['figurex_plus'], 'impact': 1.3, 'duration': 7},
            {'date': '2024-12-31', 'type': 'year_end_enterprise', 'products': ['enterprise'], 'impact': 1.6,
             'duration': 14},
        ]

        # Create comprehensive dataset
        data_rows = []

        for product in products:
            for geography in geographies:
                params = base_params[(product, geography)]

                # Base exponential growth
                base_revenue = np.array([
                    params['initial'] * (1 + params['growth']) ** (i / 30) * np.random.normal(1, 0.02)
                    # Monthly compounding
                    for i in range(n_days)
                ])

                # Add seasonality (different patterns by product)
                if product == 'figurex_plus':
                    # Consumer subscription - lower on weekends, higher during work periods
                    seasonal_pattern = np.array([
                        1 - params['seasonality'] * 0.3 if dates[i].weekday() >= 5 else 1 + params['seasonality'] * 0.1
                        for i in range(n_days)
                    ])
                elif product == 'api':
                    # Developer usage - more consistent, slight weekend dip
                    seasonal_pattern = np.array([
                        1 - params['seasonality'] * 0.1 if dates[i].weekday() >= 5 else 1
                        for i in range(n_days)
                    ])
                else:  # enterprise
                    # B2B contracts - strong quarterly patterns, Q4 spike
                    seasonal_pattern = np.array([
                        1 + params['seasonality'] * (0.5 if dates[i].month == 12 else
                                                     0.2 if dates[i].month in [3, 6, 9] else 0)
                        for i in range(n_days)
                    ])

                # Apply event impacts
                event_multipliers = np.ones(n_days)
                for event in events:
                    if product in event['products']:
                        event_date = pd.to_datetime(event['date'])
                        start_date_ts = pd.to_datetime(start_date)
                        end_date_ts = pd.to_datetime(end_date)

                        if start_date_ts <= event_date <= end_date_ts:
                            event_idx = (event_date - start_date_ts).days

                            # Different impact patterns by product
                            if product == 'enterprise':
                                # Slower ramp, longer duration for enterprise
                                for i in range(min(event['duration'], n_days - event_idx)):
                                    if event_idx + i < n_days:
                                        ramp_factor = min(1.0, i / 30)  # 30-day ramp
                                        decay_factor = np.exp(-max(0, i - 30) / (event['duration'] / 4))
                                        impact = 1 + (event['impact'] - 1) * ramp_factor * decay_factor
                                        event_multipliers[event_idx + i] *= impact
                            else:
                                # Immediate spike with decay for Plus/API
                                for i in range(min(event['duration'], n_days - event_idx)):
                                    if event_idx + i < n_days:
                                        decay_factor = np.exp(-i / (event['duration'] / 3))
                                        impact = 1 + (event['impact'] - 1) * decay_factor
                                        event_multipliers[event_idx + i] *= impact

                # Combine all effects
                final_revenue = base_revenue * seasonal_pattern * event_multipliers

                # Add product-specific adjustments
                if product == 'api':
                    # API has more volatility due to usage spikes
                    final_revenue *= np.random.normal(1, 0.05, n_days)
                elif product == 'enterprise':
                    # Enterprise is lumpier due to large deals
                    for i in range(n_days):
                        if np.random.random() < 0.02:  # 2% chance of large deal
                            final_revenue[i] *= np.random.uniform(2, 5)

                # Ensure no negative revenue
                final_revenue = np.maximum(final_revenue, params['initial'] * 0.1)

                # Create rows for this product-geography combination
                for i, date in enumerate(dates):
                    data_rows.append({
                        'date': date,
                        'product': product,
                        'geography': geography,
                        'revenue': final_revenue[i],
                        'base_revenue': base_revenue[i],
                        'seasonal_factor': seasonal_pattern[i],
                        'event_multiplier': event_multipliers[i],
                        'day_of_week': date.weekday(),
                        'month': date.month,
                        'quarter': date.quarter,
                        'is_weekend': date.weekday() >= 5,
                        'is_month_end': date.day >= 28,
                        'is_quarter_end': date.month in [3, 6, 9, 12] and date.day >= 28
                    })

        df = pd.DataFrame(data_rows)

        # Add cross-product effects
        df = self._apply_cross_product_effects(df)

        # Add external economic factors
        df = self._add_economic_factors(df)

        self.events_df = pd.DataFrame(events)
        self.events_df['date'] = pd.to_datetime(self.events_df['date'])

        return df

    def _apply_cross_product_effects(self, df):
        """Apply cross-product cannibalization and complementarity effects"""
        df_adj = df.copy()

        # Group by date and geography to apply cross-effects
        for (date, geo), group in df.groupby(['date', 'geography']):
            if len(group) == 3:  # Should have all 3 products
                plus_rev = group[group['product'] == 'figurex_plus']['revenue'].iloc[0]
                api_rev = group[group['product'] == 'api']['revenue'].iloc[0]
                enterprise_rev = group[group['product'] == 'enterprise']['revenue'].iloc[0]

                # Apply cross-effects with some randomness
                plus_effect = (1 +
                               self.cross_effects['api_to_plus'] * (api_rev / 50000) * np.random.normal(1, 0.1) +
                               self.cross_effects['enterprise_to_plus'] * (enterprise_rev / 100000) * np.random.normal(
                            1, 0.1))

                api_effect = (1 +
                              self.cross_effects['plus_to_api'] * (plus_rev / 30000) * np.random.normal(1, 0.1) +
                              self.cross_effects['enterprise_to_api'] * (enterprise_rev / 100000) * np.random.normal(1,
                                                                                                                     0.1))

                # Apply effects
                df_adj.loc[(df_adj['date'] == date) & (df_adj['geography'] == geo) &
                           (df_adj['product'] == 'figurex_plus'), 'revenue'] *= max(0.5, plus_effect)

                df_adj.loc[(df_adj['date'] == date) & (df_adj['geography'] == geo) &
                           (df_adj['product'] == 'api'), 'revenue'] *= max(0.5, api_effect)

        return df_adj

    def _add_economic_factors(self, df):
        """Add external economic factors that affect revenue"""
        df_econ = df.copy()

        # Simulate economic indicators
        dates_unique = df['date'].unique()
        n_dates = len(dates_unique)

        # GDP growth proxy (affects enterprise spending)
        gdp_growth = 2.5 + 0.5 * np.sin(np.arange(n_dates) * 2 * np.pi / 365) + np.random.normal(0, 0.3, n_dates)

        # Tech stock index proxy (affects API/developer spending)
        tech_index = 100 * (1.05 ** (np.arange(n_dates) / 365)) * (
                1 + 0.1 * np.sin(np.arange(n_dates) * 2 * np.pi / 365) + np.random.normal(0, 0.05, n_dates))

        # USD exchange rate impact (affects international revenue)
        usd_strength = 1 + 0.1 * np.sin(np.arange(n_dates) * 2 * np.pi / 365) + np.random.normal(0, 0.02, n_dates)

        # Create economic factors dataframe
        econ_df = pd.DataFrame({
            'date': dates_unique,
            'gdp_growth': gdp_growth,
            'tech_index': tech_index,
            'usd_strength': usd_strength
        })

        # Merge with main dataframe
        df_econ = df_econ.merge(econ_df, on='date', how='left')

        # Apply economic effects
        # Enterprise affected by GDP growth
        enterprise_mask = df_econ['product'] == 'enterprise'
        df_econ.loc[enterprise_mask, 'revenue'] *= (1 + (df_econ.loc[enterprise_mask, 'gdp_growth'] - 2.5) * 0.02)

        # API affected by tech index
        api_mask = df_econ['product'] == 'api'
        df_econ.loc[api_mask, 'revenue'] *= (df_econ.loc[api_mask, 'tech_index'] / 100)

        # International revenue affected by USD strength
        intl_mask = df_econ['geography'] != 'north_america'
        df_econ.loc[intl_mask, 'revenue'] /= df_econ.loc[intl_mask, 'usd_strength']

        return df_econ

    def create_features(self, df):
        """Create features for machine learning models"""
        features_df = df.copy()

        # Sort by product, geography, date for proper lag calculation
        features_df = features_df.sort_values(['product', 'geography', 'date'])

        # Lag features by product-geography
        for product in features_df['product'].unique():
            for geo in features_df['geography'].unique():
                mask = (features_df['product'] == product) & (features_df['geography'] == geo)
                if mask.sum() > 0:
                    subset = features_df[mask].copy()

                    # Lag features
                    for lag in [1, 7, 14, 30, 90]:
                        features_df.loc[mask, f'revenue_lag_{lag}'] = subset['revenue'].shift(lag)

                    # Rolling statistics
                    for window in [7, 30, 90]:
                        features_df.loc[mask, f'revenue_rolling_mean_{window}'] = subset['revenue'].rolling(
                            window).mean()
                        features_df.loc[mask, f'revenue_rolling_std_{window}'] = subset['revenue'].rolling(window).std()

                    # Growth rates
                    features_df.loc[mask, 'revenue_growth_7d'] = subset['revenue'].pct_change(7)
                    features_df.loc[mask, 'revenue_growth_30d'] = subset['revenue'].pct_change(30)

        # Time-based features
        features_df['days_since_start'] = (features_df['date'] - features_df['date'].min()).dt.days
        features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
        features_df['sin_quarter'] = np.sin(2 * np.pi * features_df['quarter'] / 4)
        features_df['cos_quarter'] = np.cos(2 * np.pi * features_df['quarter'] / 4)

        # Product and geography dummies
        features_df = pd.get_dummies(features_df, columns=['product', 'geography'], prefix=['prod', 'geo'])

        # Interaction features
        product_cols = [col for col in features_df.columns if col.startswith('prod_')]
        geo_cols = [col for col in features_df.columns if col.startswith('geo_')]

        # Create interaction terms for key combinations
        if 'prod_enterprise' in features_df.columns and 'geo_north_america' in features_df.columns:
            features_df['enterprise_na_interaction'] = (features_df['prod_enterprise'] *
                                                        features_df['geo_north_america'])

        return features_df

    def fit_hierarchical_models(self, df):
        """Fit models for each product-geography combination"""

        # Prepare features
        features_df = self.create_features(df)

        # Feature columns for ML models
        feature_cols = [col for col in features_df.columns if col not in [
            'date', 'revenue', 'base_revenue', 'seasonal_factor', 'event_multiplier'
        ]]

        # Fit models for each product-geography combination
        for product in df['product'].unique():
            for geography in df['geography'].unique():

                segment_key = f"{product}_{geography}"
                segment_data = features_df[(features_df[f'prod_{product}'] == 1) &
                                           (features_df[f'geo_{geography}'] == 1)].copy()

                if len(segment_data) < 100:  # Need sufficient data
                    continue

                # Prepare Prophet data
                prophet_data = segment_data[['date', 'revenue']].rename(columns={'date': 'ds', 'revenue': 'y'})

                # Fit Prophet model
                prophet_model = Prophet(
                    growth='linear',
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.1,
                    uncertainty_samples=1000
                )

                # Add custom seasonalities based on product type
                if product == 'enterprise':
                    prophet_model.add_seasonality(name='quarterly', period=91.25, fourier_order=4)
                elif product == 'figurex_plus':
                    prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=3)

                prophet_model.fit(prophet_data)

                # Fit XGBoost model for residuals/complex patterns
                X = segment_data[feature_cols].fillna(0)
                y = segment_data['revenue']

                # Remove any remaining NaN/inf values
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )

                try:
                    xgb_model.fit(X, y)
                    # Store feature importance
                    self.feature_importance[segment_key] = dict(zip(feature_cols, xgb_model.feature_importances_))
                except:
                    xgb_model = None

                # Store models
                self.models[segment_key] = {
                    'prophet': prophet_model,
                    'xgb': xgb_model,
                    'feature_cols': feature_cols
                }

        return self.models

    def generate_forecasts(self, df, forecast_periods=365):
        """Generate hierarchical forecasts for all segments"""

        # Generate forecasts for each segment
        segment_forecasts = {}

        for segment_key, models in self.models.items():
            product, geography = segment_key.split('_', 1)

            # Get segment data
            segment_data = df[(df['product'] == product) & (df['geography'] == geography)].copy()

            if len(segment_data) < 100:
                continue

            # Prophet forecast
            last_date = segment_data['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                         periods=forecast_periods, freq='D')

            future_df = pd.DataFrame({
                'ds': future_dates
            })

            prophet_forecast = models['prophet'].predict(future_df)

            # XGBoost forecast (if available)
            if models['xgb'] is not None:
                # Create future features matching the training data structure
                future_features_df = pd.DataFrame({
                    'date': future_dates,
                    'product': product,
                    'geography': geography,
                    'revenue': 0,  # Placeholder, will be filled by lag features
                    'day_of_week': future_dates.weekday,
                    'month': future_dates.month,
                    'quarter': future_dates.quarter,
                    'is_weekend': future_dates.weekday >= 5,
                    'is_month_end': future_dates.day >= 28,
                    'is_quarter_end': ((future_dates.month.isin([3, 6, 9, 12])) & (future_dates.day >= 28))
                })

                # Add economic features with forward fill
                last_values = segment_data.iloc[-1]
                for col in ['gdp_growth', 'tech_index', 'usd_strength']:
                    if col in segment_data.columns:
                        future_features_df[col] = last_values[col]
                    else:
                        # Add default values if not present
                        future_features_df[col] = 1.0

                # Add placeholder columns for other features
                for col in ['base_revenue', 'seasonal_factor', 'event_multiplier']:
                    if col in segment_data.columns:
                        future_features_df[col] = last_values[col]

                # Use the last known revenue as initial revenue for lag calculations
                future_features_df['revenue'] = segment_data['revenue'].iloc[-1]

                # Create full feature set by combining historical and future data
                extended_df = pd.concat([segment_data, future_features_df], ignore_index=True)
                extended_features = self.create_features(extended_df)

                # Get only the columns that were used during training
                missing_cols = [col for col in models['feature_cols'] if col not in extended_features.columns]

                # Add missing columns with zeros
                for col in missing_cols:
                    extended_features[col] = 0

                # Get future portion with correct columns in correct order
                future_X = extended_features.iloc[-forecast_periods:][models['feature_cols']].fillna(0)
                future_X = future_X.replace([np.inf, -np.inf], np.nan).fillna(0)

                try:
                    xgb_forecast = models['xgb'].predict(future_X)
                except Exception as e:
                    print(f"XGBoost prediction failed for {segment_key}: {str(e)}")
                    xgb_forecast = prophet_forecast['yhat'].values
            else:
                xgb_forecast = prophet_forecast['yhat'].values

            # Ensemble forecast (70% Prophet, 30% XGBoost)
            ensemble_forecast = 0.7 * prophet_forecast['yhat'].values + 0.3 * xgb_forecast

            # Create forecast dataframe
            segment_forecast = pd.DataFrame({
                'date': future_dates,
                'product': product,
                'geography': geography,
                'revenue_forecast': ensemble_forecast,
                'prophet_forecast': prophet_forecast['yhat'].values,
                'xgb_forecast': xgb_forecast,
                'forecast_lower': prophet_forecast['yhat_lower'].values,
                'forecast_upper': prophet_forecast['yhat_upper'].values
            })

            segment_forecasts[segment_key] = segment_forecast

        # Combine all segment forecasts
        if segment_forecasts:
            all_forecasts = pd.concat(segment_forecasts.values(), ignore_index=True)
            self.forecasts = all_forecasts
            return all_forecasts
        else:
            return pd.DataFrame()

    def create_strategic_summary(self, df, forecasts, quarters_ahead=4):
        """Create executive summary for strategic planning"""

        if forecasts.empty:
            return "No forecasts available"

        # Current revenue (last month)
        current_date = df['date'].max()
        current_month_data = df[df['date'] >= (current_date - timedelta(days=30))]
        current_revenue_monthly = current_month_data.groupby(['product', 'geography'])['revenue'].sum()
        total_current_monthly = current_revenue_monthly.sum()

        # Quarterly forecast aggregation
        forecasts_with_quarter = forecasts.copy()
        forecasts_with_quarter['quarter'] = forecasts_with_quarter['date'].dt.to_period('Q')

        quarterly_forecasts = forecasts_with_quarter.groupby(['quarter', 'product', 'geography'])[
            'revenue_forecast'].sum().reset_index()

        # Get next 4 quarters
        current_quarter = pd.Period(current_date, freq='Q')
        future_quarters = [current_quarter + i for i in range(1, quarters_ahead + 1)]

        quarterly_summary = {}
        for quarter in future_quarters:
            quarter_data = quarterly_forecasts[quarterly_forecasts['quarter'] == quarter]

            if not quarter_data.empty:
                product_breakdown = quarter_data.groupby('product')['revenue_forecast'].sum()
                geo_breakdown = quarter_data.groupby('geography')['revenue_forecast'].sum()
                total_revenue = quarter_data['revenue_forecast'].sum()

                quarterly_summary[str(quarter)] = {
                    'total': total_revenue,
                    'products': product_breakdown.to_dict(),
                    'geographies': geo_breakdown.to_dict()
                }

        # Generate executive report
        report = self._generate_executive_report(quarterly_summary, total_current_monthly)

        return report, quarterly_summary

    def _generate_executive_report(self, quarterly_summary, current_monthly_revenue):
        """Generate formatted executive report"""

        if not quarterly_summary:
            return "Insufficient data for executive report"

        # Get first quarter data for detailed breakdown
        first_quarter = list(quarterly_summary.keys())[0]
        q1_data = quarterly_summary[first_quarter]

        # Calculate growth rates
        current_quarterly = current_monthly_revenue * 3  # Approximate quarterly from monthly
        q1_growth = (q1_data['total'] - current_quarterly) / current_quarterly * 100 if current_quarterly > 0 else 0

        # Calculate year-over-year if we have enough quarters
        yoy_growth = 0
        if len(quarterly_summary) >= 4:
            q4_data = quarterly_summary[list(quarterly_summary.keys())[3]]
            yoy_growth = (q4_data[
                              'total'] - current_quarterly) / current_quarterly * 100 if current_quarterly > 0 else 0

        # Calculate hiring needs based on revenue growth
        q1_revenue_growth = q1_data['total'] - current_quarterly
        revenue_per_employee = 500000  # Assumed annual revenue per employee

        sales_hires = int(q1_data['products'].get('enterprise', 0) / revenue_per_employee * 0.2)
        engineering_hires = int(q1_data['products'].get('api', 0) / revenue_per_employee * 0.3)
        support_hires = int(q1_data['products'].get('figurex_plus', 0) / revenue_per_employee * 0.1)

        report = f"""FigureX.ai REVENUE FORECAST EXECUTIVE SUMMARY
===============================================

📊 QUARTERLY PROJECTIONS:
"""

        for quarter, data in list(quarterly_summary.items())[:4]:
            growth = (data['total'] / current_quarterly - 1) * 100 if current_quarterly > 0 else 0
            report += f"• {quarter}: ${data['total'] / 1e6:.1f}M ({growth:+.1f}%)\n"

        report += f"""
📈 PRODUCT BREAKDOWN ({first_quarter}):
• FigureX Plus: ${q1_data['products'].get('figurex_plus', 0) / 1e6:.1f}M ({q1_data['products'].get('figurex_plus', 0) / q1_data['total'] * 100:.0f}%)
• API Revenue: ${q1_data['products'].get('api', 0) / 1e6:.1f}M ({q1_data['products'].get('api', 0) / q1_data['total'] * 100:.0f}%)
• Enterprise: ${q1_data['products'].get('enterprise', 0) / 1e6:.1f}M ({q1_data['products'].get('enterprise', 0) / q1_data['total'] * 100:.0f}%)

🌍 GEOGRAPHIC BREAKDOWN ({first_quarter}):
• North America: ${q1_data['geographies'].get('north_america', 0) / 1e6:.1f}M ({q1_data['geographies'].get('north_america', 0) / q1_data['total'] * 100:.0f}%)
• Europe: ${q1_data['geographies'].get('europe', 0) / 1e6:.1f}M ({q1_data['geographies'].get('europe', 0) / q1_data['total'] * 100:.0f}%)
• APAC: ${q1_data['geographies'].get('apac', 0) / 1e6:.1f}M ({q1_data['geographies'].get('apac', 0) / q1_data['total'] * 100:.0f}%)
• Other: ${q1_data['geographies'].get('other', 0) / 1e6:.1f}M ({q1_data['geographies'].get('other', 0) / q1_data['total'] * 100:.0f}%)

🎯 KEY GROWTH DRIVERS:
• Enterprise segment showing strongest growth trajectory
• APAC region outpacing mature markets in growth rate
• API revenue benefiting from developer ecosystem expansion
• Cross-product synergies driving customer lifetime value

🧑‍💼 HIRING IMPLICATIONS:
• Sales Team: +{sales_hires} headcount (Enterprise expansion)
• Engineering: +{engineering_hires} headcount (API & infrastructure scaling)
• Customer Success: +{support_hires} headcount (FigureX Plus support)
• Total New Hires: ~{sales_hires + engineering_hires + support_hires} across all functions

💡 STRATEGIC RECOMMENDATIONS:
1. Accelerate enterprise sales motion in high-growth regions (APAC, Europe)
2. Invest in API infrastructure to support {(q1_data['products'].get('api', 0) / current_quarterly * 3 - 1) * 100:.0f}% growth
3. Enhance FigureX Plus features to reduce churn and increase ARPU
4. Establish regional data centers in APAC to capture market opportunity
5. Develop verticalized enterprise solutions for key industries

⚠️ RISK FACTORS:
• Competitive pressure from emerging LLM providers
• Regulatory uncertainty in key markets (EU, China)
• Infrastructure scaling challenges at projected growth rates
• Talent acquisition in competitive AI/ML hiring market

📅 NEXT STEPS:
• Q1: Scale sales team and launch enterprise vertical solutions
• Q2: Expand APAC infrastructure and partnerships
• Q3: Release next-generation API features
• Q4: Optimize pricing and packaging across all products

Year-over-Year Growth Projection: {yoy_growth:+.1f}%

💰 FINANCIAL SUMMARY:
• Current Monthly Run Rate: ${current_monthly_revenue / 1e6:.1f}M
• Projected Q1 Revenue: ${q1_data['total'] / 1e6:.1f}M
• Projected Annual Revenue: ${sum(data['total'] for data in quarterly_summary.values()) / 1e6:.1f}M
• Implied Valuation (30x Revenue): ${sum(data['total'] for data in quarterly_summary.values()) * 30 / 1e9:.1f}B

🚀 MOMENTUM INDICATORS:
• Quarter-over-Quarter Growth: {q1_growth:.1f}%
• Product Mix Shift: {"Enterprise-focused" if q1_data['products'].get('enterprise', 0) / q1_data['total'] > 0.4 else "Balanced across segments"}
• Geographic Expansion: {"Strong international growth" if (q1_data['geographies'].get('europe', 0) + q1_data['geographies'].get('apac', 0) + q1_data['geographies'].get('other', 0)) / q1_data['total'] > 0.5 else "NA-dominant"}
• Revenue Velocity: ${(q1_data['total'] - current_quarterly) / 90 / 1e6:.2f}M/day incremental

This forecast assumes continued market expansion, successful product launches, and stable competitive dynamics.
"""

        return report

    def calculate_scenario_analysis(self, df, forecasts, scenarios=None):
        """Calculate different scenario projections"""

        if scenarios is None:
            scenarios = {
                'base': {'growth_multiplier': 1.0, 'description': 'Base case forecast'},
                'bull': {'growth_multiplier': 1.3, 'description': 'Accelerated AI adoption'},
                'bear': {'growth_multiplier': 0.7, 'description': 'Increased competition'}
            }

        scenario_results = {}

        for scenario_name, scenario_params in scenarios.items():
            scenario_forecast = forecasts.copy()
            scenario_forecast['revenue_forecast'] *= scenario_params['growth_multiplier']
            scenario_forecast['forecast_lower'] *= scenario_params['growth_multiplier']
            scenario_forecast['forecast_upper'] *= scenario_params['growth_multiplier']

            # Calculate quarterly totals
            scenario_forecast['quarter'] = scenario_forecast['date'].dt.to_period('Q')
            quarterly_totals = scenario_forecast.groupby('quarter')['revenue_forecast'].sum()

            scenario_results[scenario_name] = {
                'description': scenario_params['description'],
                'multiplier': scenario_params['growth_multiplier'],
                'quarterly_revenue': quarterly_totals.to_dict(),
                'annual_revenue': quarterly_totals.sum()
            }

        return scenario_results

    def export_results(self, df, forecasts, filename_prefix='figurex_revenue'):
        """Export historical data and forecasts to CSV files"""

        # Export historical data
        historical_filename = f"2revenue_forecast/{filename_prefix}_historical.csv"
        df.to_csv(historical_filename, index=False)

        # Export forecasts
        forecast_filename = f"2revenue_forecast/{filename_prefix}_forecast.csv"
        forecasts.to_csv(forecast_filename, index=False)

        # Export aggregated summary
        summary_data = pd.concat([
            df.groupby(['date', 'product'])['revenue'].sum().reset_index(),
            forecasts.groupby(['date', 'product'])['revenue_forecast'].sum().reset_index()
            .rename(columns={'revenue_forecast': 'revenue'})
        ])
        summary_filename = f"2revenue_forecast/{filename_prefix}_summary.csv"
        summary_data.to_csv(summary_filename, index=False)

        print(f"Results exported to:")
        print(f"  - {historical_filename}")
        print(f"  - {forecast_filename}")
        print(f"  - {summary_filename}")

        return {
            'historical': historical_filename,
            'forecast': forecast_filename,
            'summary': summary_filename
        }

    def plot_results(self, df, forecasts, save_plots=True, plot_prefix='figurex_revenue'):
        """Create comprehensive visualization of results"""

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Overall Revenue Trend with Forecast
        ax1 = fig.add_subplot(gs[0, :])

        # Aggregate historical data
        historical_daily = df.groupby('date')['revenue'].sum().reset_index()
        forecast_daily = forecasts.groupby('date')['revenue_forecast'].sum().reset_index()

        ax1.plot(historical_daily['date'], historical_daily['revenue'] / 1e6,
                 label='Historical', color=colors[0], linewidth=2)
        ax1.plot(forecast_daily['date'], forecast_daily['revenue_forecast'] / 1e6,
                 label='Forecast', color=colors[1], linewidth=2, linestyle='--')

        # Add uncertainty bands
        forecast_bounds = forecasts.groupby('date')[['forecast_lower', 'forecast_upper']].sum()
        ax1.fill_between(forecast_bounds.index,
                         forecast_bounds['forecast_lower'] / 1e6,
                         forecast_bounds['forecast_upper'] / 1e6,
                         alpha=0.2, color=colors[1], label='Uncertainty')

        ax1.set_title('FigureX.ai Total Revenue: Historical & Forecast', fontsize=16, pad=20)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Revenue ($M)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Revenue by Product
        ax2 = fig.add_subplot(gs[1, 0])

        product_historical = df.groupby(['date', 'product'])['revenue'].sum().reset_index()
        product_forecast = forecasts.groupby(['date', 'product'])['revenue_forecast'].sum().reset_index()

        for i, product in enumerate(['figurex_plus', 'api', 'enterprise']):
            hist_data = product_historical[product_historical['product'] == product]
            fore_data = product_forecast[product_forecast['product'] == product]

            ax2.plot(hist_data['date'], hist_data['revenue'] / 1e6,
                     color=colors[i], linewidth=2, label=f'{product} (hist)')
            ax2.plot(fore_data['date'], fore_data['revenue_forecast'] / 1e6,
                     color=colors[i], linewidth=2, linestyle='--', alpha=0.7)

        ax2.set_title('Revenue by Product', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Revenue ($M)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. Revenue by Geography
        ax3 = fig.add_subplot(gs[1, 1])

        geo_historical = df.groupby(['date', 'geography'])['revenue'].sum().reset_index()
        geo_forecast = forecasts.groupby(['date', 'geography'])['revenue_forecast'].sum().reset_index()

        for i, geo in enumerate(['north_america', 'europe', 'apac', 'other']):
            hist_data = geo_historical[geo_historical['geography'] == geo]
            fore_data = geo_forecast[geo_forecast['geography'] == geo]

            ax3.plot(hist_data['date'], hist_data['revenue'] / 1e6,
                     color=colors[i], linewidth=2, label=f'{geo} (hist)')
            ax3.plot(fore_data['date'], fore_data['revenue_forecast'] / 1e6,
                     color=colors[i], linewidth=2, linestyle='--', alpha=0.7)

        ax3.set_title('Revenue by Geography', fontsize=14)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Revenue ($M)')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 4. Product Mix Evolution
        ax4 = fig.add_subplot(gs[2, 0])

        # Calculate monthly product mix
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['date'].dt.to_period('M')
        product_mix = df_monthly.groupby(['month', 'product'])['revenue'].sum().unstack(fill_value=0)
        product_mix_pct = product_mix.div(product_mix.sum(axis=1), axis=0) * 100

        # Stack plot
        ax4.stackplot(product_mix_pct.index.astype(str),
                      product_mix_pct['figurex_plus'],
                      product_mix_pct['api'],
                      product_mix_pct['enterprise'],
                      labels=['FigureX Plus', 'API', 'Enterprise'],
                      colors=colors[:3],
                      alpha=0.8)

        ax4.set_title('Product Mix Evolution (%)', fontsize=14)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Percentage of Total Revenue')
        ax4.legend(loc='upper right')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.grid(True, alpha=0.3)

        # 5. Geographic Mix Evolution
        ax5 = fig.add_subplot(gs[2, 1])

        geo_mix = df_monthly.groupby(['month', 'geography'])['revenue'].sum().unstack(fill_value=0)
        geo_mix_pct = geo_mix.div(geo_mix.sum(axis=1), axis=0) * 100

        ax5.stackplot(geo_mix_pct.index.astype(str),
                      geo_mix_pct['north_america'],
                      geo_mix_pct['europe'],
                      geo_mix_pct['apac'],
                      geo_mix_pct['other'],
                      labels=['North America', 'Europe', 'APAC', 'Other'],
                      colors=colors[:4],
                      alpha=0.8)

        ax5.set_title('Geographic Mix Evolution (%)', fontsize=14)
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Percentage of Total Revenue')
        ax5.legend(loc='upper right')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
        ax5.grid(True, alpha=0.3)

        # 6. Growth Rates by Product
        ax6 = fig.add_subplot(gs[3, 0])

        # Calculate month-over-month growth rates
        for product in ['figurex_plus', 'api', 'enterprise']:
            product_monthly = product_mix[product]
            growth_rate = product_monthly.pct_change() * 100
            ax6.plot(growth_rate.index.astype(str)[1:], growth_rate[1:],
                     marker='o', label=product, linewidth=2)

        ax6.set_title('Month-over-Month Growth Rate by Product', fontsize=14)
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Growth Rate (%)')
        ax6.legend()
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
        ax6.grid(True, alpha=0.3)

        # 7. Quarterly Revenue Projections
        ax7 = fig.add_subplot(gs[3, 1])

        # Aggregate to quarterly
        df_quarterly = df.copy()
        df_quarterly['quarter'] = df_quarterly['date'].dt.to_period('Q')
        quarterly_hist = df_quarterly.groupby('quarter')['revenue'].sum() / 1e6

        forecasts_quarterly = forecasts.copy()
        forecasts_quarterly['quarter'] = forecasts_quarterly['date'].dt.to_period('Q')
        quarterly_fore = forecasts_quarterly.groupby('quarter')['revenue_forecast'].sum() / 1e6

        # Combine for plotting
        quarters = list(quarterly_hist.index.astype(str)) + list(quarterly_fore.index.astype(str))
        revenues = list(quarterly_hist.values) + list(quarterly_fore.values)
        colors_bar = ['#1f77b4'] * len(quarterly_hist) + ['#ff7f0e'] * len(quarterly_fore)

        bars = ax7.bar(range(len(quarters)), revenues, color=colors_bar)
        ax7.set_xticks(range(len(quarters)))
        ax7.set_xticklabels(quarters, rotation=45)
        ax7.set_title('Quarterly Revenue (Historical & Forecast)', fontsize=14)
        ax7.set_xlabel('Quarter')
        ax7.set_ylabel('Revenue ($M)')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height,
                     f'${height:.0f}M', ha='center', va='bottom')

        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Feature Importance (if available)
        ax8 = fig.add_subplot(gs[4, :])

        if self.feature_importance:
            # Get top features across all segments
            all_features = {}
            for segment, features in self.feature_importance.items():
                for feature, importance in features.items():
                    if feature not in all_features:
                        all_features[feature] = 0
                    all_features[feature] += importance

            # Sort and get top 20
            top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_names = [f[0] for f in top_features]
            feature_values = [f[1] for f in top_features]

            bars = ax8.barh(range(len(feature_names)), feature_values, color=colors[0])
            ax8.set_yticks(range(len(feature_names)))
            ax8.set_yticklabels(feature_names)
            ax8.set_title('Top 20 Most Important Features (XGBoost)', fontsize=14)
            ax8.set_xlabel('Cumulative Feature Importance')
            ax8.grid(True, alpha=0.3, axis='x')
        else:
            ax8.text(0.5, 0.5, 'Feature importance not available',
                     ha='center', va='center', transform=ax8.transAxes)

        plt.suptitle('FigureX.ai Revenue Analysis Dashboard', fontsize=20, y=0.995)

        if save_plots:
            plt.savefig(f'2revenue_forecast/{plot_prefix}_dashboard.png', dpi=300, bbox_inches='tight')
            print(f"Dashboard saved as {plot_prefix}_dashboard.png")

        plt.show()

        # Additional individual plots
        self._plot_scenario_analysis(df, forecasts, save_plots, plot_prefix)
        self._plot_segment_details(df, forecasts, save_plots, plot_prefix)

    def _plot_scenario_analysis(self, df, forecasts, save_plots=True, plot_prefix='figurex_revenue'):
        """Plot scenario analysis results"""

        # Run scenario analysis
        scenarios = self.calculate_scenario_analysis(df, forecasts)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Quarterly projections by scenario
        scenario_names = list(scenarios.keys())
        quarters = list(list(scenarios.values())[0]['quarterly_revenue'].keys())

        x = np.arange(len(quarters))
        width = 0.25

        for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
            quarterly_values = [scenario_data['quarterly_revenue'][q] / 1e6 for q in quarters]
            ax1.bar(x + i * width, quarterly_values, width, label=scenario_name.capitalize())

        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Revenue ($M)')
        ax1.set_title('Quarterly Revenue by Scenario')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([str(q) for q in quarters], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Annual revenue comparison
        annual_revenues = [data['annual_revenue'] / 1e6 for data in scenarios.values()]
        colors_scenario = ['#2ca02c', '#1f77b4', '#d62728']

        bars = ax2.bar(scenario_names, annual_revenues, color=colors_scenario)
        ax2.set_ylabel('Annual Revenue ($M)')
        ax2.set_title('Annual Revenue Forecast by Scenario')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'${height:.0f}M', ha='center', va='bottom')

        plt.suptitle('Scenario Analysis', fontsize=16)

        if save_plots:
            plt.savefig(f'2revenue_forecast/{plot_prefix}_scenarios.png', dpi=300, bbox_inches='tight')
            print(f"Scenario analysis saved as {plot_prefix}_scenarios.png")

        plt.show()

    def _plot_segment_details(self, df, forecasts, save_plots=True, plot_prefix='figurex_revenue'):
        """Plot detailed view of top revenue segments"""

        # Get top 6 segments by revenue
        segment_revenue = df.groupby(['product', 'geography'])['revenue'].sum().sort_values(ascending=False)
        top_segments = segment_revenue.head(6).index

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (product, geography) in enumerate(top_segments):
            ax = axes[idx]

            # Historical data
            hist_data = df[(df['product'] == product) & (df['geography'] == geography)]
            hist_monthly = hist_data.groupby(hist_data['date'].dt.to_period('M'))['revenue'].sum() / 1e6

            # Forecast data
            fore_data = forecasts[(forecasts['product'] == product) & (forecasts['geography'] == geography)]
            if not fore_data.empty:
                fore_monthly = fore_data.groupby(fore_data['date'].dt.to_period('M'))['revenue_forecast'].sum() / 1e6

                # Plot
                ax.plot(hist_monthly.index.astype(str), hist_monthly.values,
                        label='Historical', marker='o', linewidth=2)
                ax.plot(fore_monthly.index.astype(str), fore_monthly.values,
                        label='Forecast', marker='s', linestyle='--', linewidth=2)
            else:
                ax.plot(hist_monthly.index.astype(str), hist_monthly.values,
                        label='Historical', marker='o', linewidth=2)

            ax.set_title(f'{product.title()} - {geography.title()}')
            ax.set_xlabel('Month')
            ax.set_ylabel('Revenue ($M)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
        plt.suptitle('Top Revenue Segments: Historical & Forecast', fontsize=16)
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'2revenue_forecast/{plot_prefix}_segments.png', dpi=300, bbox_inches='tight')
            print(f"Segment details saved as {plot_prefix}_segments.png")

        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = RevenueForecaster()

    # Generate synthetic revenue data
    print("Generating synthetic revenue data...")
    revenue_data = forecaster.generate_synthetic_revenue_data(
        start_date='2022-11-01',
        end_date='2024-12-31'
    )

    # Fit hierarchical models
    print("\nFitting forecasting models...")
    models = forecaster.fit_hierarchical_models(revenue_data)
    print(f"Fitted models for {len(models)} product-geography segments")

    # Generate forecasts
    print("\nGenerating revenue forecasts...")
    forecasts = forecaster.generate_forecasts(revenue_data, forecast_periods=365)

    # Create strategic summary
    print("\nCreating executive summary...")
    report, quarterly_summary = forecaster.create_strategic_summary(
        revenue_data,
        forecasts,
        quarters_ahead=4
    )

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Run scenario analysis
    print("\nRunning scenario analysis...")
    scenarios = forecaster.calculate_scenario_analysis(revenue_data, forecasts)

    print("\nSCENARIO ANALYSIS RESULTS:")
    for scenario_name, results in scenarios.items():
        print(f"\n{scenario_name.upper()} CASE: {results['description']}")
        print(f"Annual Revenue Projection: ${results['annual_revenue'] / 1e6:.1f}M")

    # Export results
    print("\nExporting results...")
    exported_files = forecaster.export_results(revenue_data, forecasts)

    # Create visualizations
    print("\nGenerating visualizations...")
    forecaster.plot_results(revenue_data, forecasts, save_plots=True)

    # Display feature importance for a sample segment
    if forecaster.feature_importance:
        sample_segment = list(forecaster.feature_importance.keys())[0]
        top_features = sorted(
            forecaster.feature_importance[sample_segment].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        print(f"\nTop 10 Features for {sample_segment}:")
        for feature, importance in top_features:
            print(f"  - {feature}: {importance:.4f}")
