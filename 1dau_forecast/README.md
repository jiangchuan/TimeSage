# 🧠 FigureX.ai DAU Forecasting Solution

A comprehensive time series forecasting system designed to predict Daily Active Users (DAU) for rapidly growing AI platforms like FigureX.ai. This solution handles complex growth patterns including viral spikes, product launches, seasonal effects, and uncertainty quantification.

## 🎯 Problem Statement

FigureX.ai experiences rapid DAU growth with sharp spikes after major product launches and media events. Traditional forecasting methods struggle with:
- Event-driven growth patterns
- Exponential base growth trends
- Seasonal variations
- High uncertainty in volatile growth phases

## 🚀 Solution Overview

Our ensemble approach combines multiple forecasting techniques:

- **Prophet Model**: Handles trend decomposition, seasonality, and planned events
- **XGBoost Model**: Captures complex feature interactions and non-linear patterns
- **Ensemble Method**: Combines both models with optimized weights
- **Uncertainty Quantification**: Provides confidence intervals for business planning

## 📊 Key Features

### 🔮 Forecasting Capabilities
- **90-day DAU predictions** with confidence intervals
- **Event impact modeling** for product launches and viral moments
- **Seasonal pattern detection** (weekly, monthly, yearly)
- **Growth trend analysis** with changepoint detection

### 📈 Advanced Analytics
- **Cross-validation** with time series splits
- **Feature importance** analysis for model interpretability
- **Residual analysis** for model validation
- **Growth trajectory comparison** against simple exponential models

### 📊 Visualization Suite
- Historical data with event markers
- Forecast plots with uncertainty bands
- Event impact analysis
- Feature importance charts
- Seasonality patterns
- Model performance metrics

## 🛠️ Installation

```bash
# Required packages
pip install pandas numpy matplotlib prophet xgboost scikit-learn scipy
```

### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
prophet>=1.1.0
xgboost>=1.5.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

## 📝 Usage

### Basic Forecasting

```python
from growth_forecaster import GrowthForecaster

# Initialize forecaster
forecaster = GrowthForecaster()

# Generate or load your data
df = forecaster.generate_synthetic_data(
    start_date='2022-11-01', 
    end_date='2024-01-31'
)

# Train models
forecaster.fit_prophet_model(df)
forecaster.fit_xgb_model(df)

# Generate 90-day forecast
forecast = forecaster.ensemble_predict(df, future_periods=90)

# Create visualizations
fig = forecaster.plot_forecast(df, forecast)

# Generate business report
report = forecaster.generate_business_report(df, forecast)
print(report)
```

### Advanced Usage

```python
# Custom ensemble weights
forecaster.ensemble_weights = {'prophet': 0.6, 'xgb': 0.4}

# Cross-validation
cv_results = forecaster.cross_validate_model(df, n_splits=5)
print(f"Average MAPE: {cv_results['mape'].mean():.2%}")

# Feature importance analysis
top_features = sorted(
    forecaster.feature_importance.items(), 
    key=lambda x: x[1], 
    reverse=True
)[:10]
```

## 📁 Data Requirements

### Input Data Format
Your DataFrame should contain:
```python
df = pd.DataFrame({
    'ds': pd.date_range('2022-01-01', periods=365),  # Date column
    'y': [1000000, 1008000, ...],                    # DAU values
    'event_viral_launch': [1, 1, 0, 0, ...],        # Event indicators (optional)
    'event_product_launch': [0, 0, 1, 1, ...],      # Event indicators (optional)
})
```

### Event Data Structure
```python
events = [
    {
        'date': '2023-03-14',
        'type': 'fx4_launch',
        'magnitude': 2.5,  # Impact multiplier
        'duration': 10     # Days of impact
    }
]
```

## 🏗️ Model Architecture

### Prophet Model
- **Growth**: Logistic growth with capacity constraints
- **Seasonality**: Daily, weekly, and yearly patterns
- **Events**: Custom holidays for product launches
- **Changepoints**: Automatic trend change detection

### XGBoost Model
- **Features**: Lag values, rolling statistics, growth rates, event indicators
- **Hyperparameters**: Optimized for time series forecasting
- **Feature Engineering**: Automated creation of 20+ predictive features

### Ensemble Method
- **Default Weights**: 70% Prophet, 30% XGBoost
- **Rationale**: Prophet handles trend/seasonality, XGBoost captures complex interactions
- **Uncertainty**: Conservative lower bounds, optimistic upper bounds

## 📊 Synthetic Data Generation

The solution includes realistic synthetic data generation that simulates:

### Growth Patterns
- **Base Growth**: 0.8% daily growth rate (realistic for AI platforms)
- **Initial Scale**: 1M DAUs (typical for successful AI launch)
- **Viral Events**: 2-3x impact multipliers with exponential decay

### Event Timeline (Example)
- **Dec 2022**: Initial viral launch (3.0x impact, 14 days)
- **Mar 2023**: FX-4 model launch (2.5x impact, 10 days)
- **Nov 2023**: FXS launch (2.2x impact, 12 days)
- **Seasonal**: Weekend usage reduction (5% lower)

## 📈 Performance Metrics

### Cross-Validation Results
- **MAPE**: 8-15% (typical for growth forecasting)
- **RMSE**: Varies with scale, reported relative to mean
- **Time Series Splits**: 5-fold validation preserving temporal order

### Model Interpretability
- **Feature Importance**: XGBoost provides ranked feature contributions
- **Trend Components**: Prophet decomposes trend, seasonal, and event effects
- **Residual Analysis**: Validates model assumptions and bias

## 📊 Output Formats

### Forecast DataFrame
```python
forecast.columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
# ds: Future dates
# yhat: Point predictions
# yhat_lower: 5th percentile confidence bound
# yhat_upper: 95th percentile confidence bound
```

### Business Report
Executive summary including:
- Current state metrics
- 3-month growth projections
- Confidence intervals
- Key insights and recommendations
- Risk assessment

## 🎨 Visualization Gallery

### Main Forecast Plot
- Historical DAU with event markers
- Forecast line with confidence intervals
- Event impact annotations

### Detailed Analysis
- **Event Impact**: Multiplier effects over time
- **Growth Rates**: 7-day and 30-day rolling rates
- **Seasonality**: Weekly usage patterns
- **Uncertainty**: Forecast confidence evolution

### Model Diagnostics
- **Residuals**: Prediction errors vs fitted values
- **Feature Importance**: Top contributing variables
- **Growth Comparison**: ML forecast vs simple exponential

## ⚠️ Limitations & Assumptions

### Model Limitations
- **Event Prediction**: Cannot predict unplanned viral events
- **Competitive Response**: Assumes no major competitive disruptions
- **Market Saturation**: Limited modeling of user acquisition limits
- **Data Requirements**: Needs 6+ months of historical data for reliability

### Key Assumptions
- **Growth Sustainability**: Current patterns continue
- **Event Patterns**: Historical event impacts predict future impacts
- **Seasonality Stability**: Usage patterns remain consistent
- **Infrastructure**: No technical constraints on user growth

## 🔧 Customization Options

### Model Parameters
```python
# Prophet customization
prophet_model = Prophet(
    growth='logistic',
    changepoint_prior_scale=0.05,  # Trend flexibility
    seasonality_prior_scale=10,    # Seasonality strength
    uncertainty_samples=1000       # Confidence interval precision
)

# XGBoost customization
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
)
```

### Feature Engineering
- **Lag Features**: 1, 7, 14, 30-day lags
- **Rolling Windows**: Mean and standard deviation over multiple periods
- **Growth Rates**: Short and long-term percentage changes
- **Event Features**: Binary indicators and days-since counters

## 📚 References & Methodology

### Forecasting Approaches
- **Prophet**: Facebook's time series forecasting tool
- **XGBoost**: Gradient boosting for non-linear patterns
- **Ensemble Methods**: Combining multiple model strengths

### Best Practices
- **Time Series CV**: Prevents data leakage in validation
- **Feature Engineering**: Domain-specific predictors for growth
- **Uncertainty Quantification**: Critical for business planning
- **Event Modeling**: Incorporating known business drivers

## 🤝 Contributing

### Development Setup
```bash
git clone [repository-url]
cd figurexa-forecasting
pip install -r requirements.txt
python -m pytest tests/
```

### Adding New Features
1. **Event Types**: Add new event categories in `generate_synthetic_data()`
2. **Models**: Implement new forecasting algorithms in ensemble
3. **Features**: Extend `create_features()` method
4. **Visualizations**: Add plots in `_plot_detailed_analysis()`


## 📄 License

MIT License

---

**Built for rapid growth analysis at scale** 🚀