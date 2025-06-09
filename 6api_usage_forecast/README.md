# API Usage Forecast During Model Transition

A comprehensive forecasting system for predicting API usage patterns during major model transitions, specifically designed for analyzing the migration from FigureX-3.5-turbo to FigureX-4o.

## Overview

This system helps organizations understand and predict how API usage will shift during model deprecation and transition periods. It combines advanced time series analysis, changepoint detection, and machine learning to provide actionable insights for infrastructure planning.

## Key Features

### 🔍 Advanced Analytics
- **Time Series Decomposition**: Breaks down usage patterns by model, seasonality, and trends
- **Changepoint Detection**: Automatically identifies significant shifts in usage patterns using ruptures library
- **Multi-Model Forecasting**: Combines Prophet and XGBoost for robust predictions
- **Scenario Planning**: Generates multiple migration scenarios (slow, aggressive, baseline)

### 📊 Comprehensive Visualizations
- Usage patterns by model over time
- Market share evolution during transition
- Changepoint identification and analysis
- Forecast uncertainty intervals
- Scenario comparison charts

### ⚡ Infrastructure Planning
- Compute requirement calculations based on model complexity
- Scaling recommendations for different scenarios
- Cost optimization insights
- Early warning system design

## Installation

```bash
# Core dependencies
pip install numpy pandas matplotlib
pip install prophet xgboost scikit-learn
pip install ruptures  # For changepoint detection

# Optional: Enhanced plotting
pip install seaborn plotly
```

## Quick Start

```python
from api_usage_forecast import APIUsageForecast

# Initialize the forecasting system
forecaster = APIUsageForecast()

# Run complete analysis pipeline
results = forecaster.run_complete_analysis()

# Access individual components
data = results['data']
forecasts = results['forecasts'] 
scenarios = results['scenarios']
compute_requirements = results['compute_requirements']
```

## Core Components

### 1. Data Generation and Analysis
```python
# Generate realistic sample data
data = forecaster.generate_sample_data(days=365)

# Analyze usage patterns by model
data_with_analysis = forecaster.decompose_usage_by_model(data)
```

### 2. Changepoint Detection
```python
# Detect significant shifts in usage patterns
changepoints = forecaster.detect_changepoints(data)

# Changepoints inform model building and scenario creation
```

### 3. Multi-Model Forecasting
```python
# Build Prophet models with external regressors
prophet_model = forecaster.build_prophet_model(data, 'total_usage')

# Build XGBoost models with feature engineering
xgb_model, features = forecaster.build_xgboost_model(data, 'total_usage')

# Generate 90-day forecasts
forecasts = forecaster.generate_forecasts(data, forecast_days=90)
```

### 4. Scenario Planning
```python
# Create migration scenarios based on changepoint analysis
scenarios = forecaster.create_scenarios(data, forecast_days=90)

# Visualize scenarios and compute requirements
compute_reqs = forecaster.visualize_scenarios(data)
```

## Model Features

### Prophet Model
- **Seasonality**: Daily, weekly, and yearly patterns
- **External Regressors**: Pricing impact, media coverage, weekend effects
- **Changepoints**: Incorporates detected usage shifts
- **Uncertainty**: Built-in confidence intervals

### XGBoost Model
- **Feature Engineering**: Lag features, moving averages, cyclical encoding
- **Changepoint Features**: Binary indicators and time-since-changepoint
- **Launch Effects**: Post-launch adoption indicators
- **Iterative Forecasting**: Multi-step ahead predictions

## External Factors

The system incorporates several external factors that influence API usage:

- **Pricing Changes**: Impact on adoption rates
- **Media Coverage**: Launch buzz and publicity effects  
- **Documentation**: Quality and availability of resources
- **Seasonality**: Day-of-week and monthly patterns
- **Launch Timeline**: Pre/post-launch dynamics

## Scenario Types

### Baseline Scenario
- Moderate migration pace based on historical patterns
- Standard adoption curves
- Balanced infrastructure scaling

### Slow Migration Scenario  
- Conservative user adoption
- Extended transition period
- Lower infrastructure pressure

### Aggressive Switching Scenario
- Rapid model adoption
- Accelerated deprecation timeline
- High infrastructure scaling needs

## Output Analysis

### Key Metrics
- Current usage distribution by model
- Growth rates and trend analysis
- Market share evolution
- Peak usage identification
- Weekend vs weekday patterns

### Infrastructure Recommendations
- Compute scaling requirements (40-80% for aggressive scenarios)
- Dynamic scaling triggers
- Cost optimization strategies
- Monitoring and alerting setup

### Executive Summary
Automatically generated summary including:
- Current state analysis
- Recent trend identification  
- Changepoint insights
- Scenario projections
- Infrastructure recommendations
- Next steps and action items

## File Structure

```
api_usage_forecast/
├── APIUsageForecast.py          # Main class
├── README.md                    # This file
├── requirements.txt             # Dependencies
└── outputs/
    ├── API_usage_by_model.png   # Usage decomposition
    ├── changepoint_detection.png # Changepoint analysis
    ├── API_usage_forecast.png   # Scenario forecasts
    └── forecast_uncertainty_analysis.png # Uncertainty
```

## Advanced Usage

### Custom Data Integration
```python
# Use your own data instead of generated samples
custom_data = pd.read_csv('your_api_usage_data.csv')

# Ensure required columns: date, figurex35_usage, figurex4o_usage, etc.
forecaster.decompose_usage_by_model(custom_data)
```

### Model Customization
```python
# Adjust Prophet parameters
model = Prophet(
    changepoint_prior_scale=0.2,  # Higher for more flexible changepoints
    seasonality_prior_scale=15,    # Higher for stronger seasonality
    yearly_seasonality=False       # Disable if not relevant
)

# Customize XGBoost features
feature_cols = ['pricing_impact', 'media_coverage', 'custom_feature']
```

### Scenario Customization
```python
# Create custom scenarios with specific parameters
custom_scenarios = {
    'conservative': (0.005, 0.05),  # (decline_rate, growth_rate)
    'moderate': (0.02, 0.15),
    'aggressive': (0.08, 0.30)
}
```

## Business Applications

### Product Management
- Model deprecation timeline planning
- Feature adoption tracking
- User migration analysis

### Infrastructure Planning  
- Capacity planning and scaling
- Cost forecasting and budgeting
- Resource allocation optimization

### Strategic Planning
- Market transition analysis
- Competitive positioning
- Investment prioritization

## Performance Considerations

### Compute Requirements
- **FigureX-3.5-turbo**: 1.0x base compute (reference)
- **FigureX-4o**: 2.5x base compute (more intensive)
- **Scaling Factor**: 40-80% additional capacity for peak scenarios

### Monitoring Recommendations
- Real-time usage dashboards
- Automated changepoint detection
- Weekly migration rate tracking
- Capacity utilization alerts

## Troubleshooting

### Common Issues

**Missing ruptures package:**
```bash
pip install ruptures
```

**Memory issues with large datasets:**
- Reduce sample size in `generate_sample_data()`
- Use data chunking for very large datasets
- Consider using Dask for distributed processing

**Poor forecast accuracy:**
- Check for data quality issues
- Adjust changepoint detection sensitivity
- Include additional external regressors
- Extend historical data period

### Model Validation
```python
# Check model performance
from sklearn.metrics import mean_absolute_error

# Split data for validation
train_data = data[:-30]  # All but last 30 days
test_data = data[-30:]   # Last 30 days

# Train on subset and validate
model_performance = forecaster.validate_models(train_data, test_data)
```

## Contributing

### Adding New Features
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-analysis`
3. Add comprehensive tests
4. Submit pull request with detailed description

### Feature Requests
Common enhancement areas:
- Additional ML models (LSTM, ARIMA)
- Real-time data integration
- Multi-region analysis
- Cost optimization algorithms

## License

MIT License - see LICENSE file for details.

---

**Built for production-scale API usage forecasting and infrastructure planning.**