# 💡 Customer Lifetime Value (LTV) Forecasting

A comprehensive customer lifetime value forecasting system that combines survival analysis, machine learning, and cohort-based modeling to predict customer value and optimize pricing strategies. This solution is designed for subscription-based and SaaS businesses looking to make data-driven pricing and acquisition decisions.

## 🎯 Business Problem

Customer Lifetime Value forecasting is critical for:
- **Pricing optimization** across different customer segments
- **Customer acquisition budgeting** with proper CAC/LTV ratios
- **Product strategy** decisions based on customer value
- **Revenue forecasting** and financial planning
- **Churn prevention** by identifying at-risk high-value customers

Traditional approaches fail to capture:
- **Customer heterogeneity** across acquisition channels and segments
- **Uncertainty quantification** for risk management
- **Survival dynamics** that affect customer lifetime
- **Revenue evolution** patterns over customer lifecycle

## 🚀 Solution Overview

Our probabilistic LTV forecasting system provides:

### 🧠 Multi-Model Architecture
- **Survival Analysis**: Kaplan-Meier, Weibull, and Log-Normal models for churn prediction
- **Revenue Forecasting**: Random Forest and Gradient Boosting for monthly revenue prediction
- **Cohort Analysis**: Segment-specific LTV patterns and trends
- **Monte Carlo Simulation**: Uncertainty quantification with confidence intervals

### 📊 Comprehensive Customer Modeling
- **Acquisition Channel Analysis**: LTV by organic, paid, social, referral sources
- **Plan Type Optimization**: Free, Basic, Pro, Enterprise value assessment
- **Company Size Segmentation**: SMB to Enterprise customer value patterns
- **Temporal Cohorts**: Time-based signup cohort performance tracking

### 💰 Pricing Strategy Support
- **Price Elasticity Analysis**: Demand sensitivity modeling
- **LTV/CAC Optimization**: Return on acquisition investment
- **Scenario Planning**: Conservative, base, and optimistic forecasts
- **Risk-Adjusted Planning**: P25/P75 confidence intervals for decision making

## 🏗️ Key Features

### 🔮 Advanced Forecasting
- **Probabilistic LTV estimates** with P10-P90 confidence intervals
- **6-24 month forward-looking** projections
- **Customer-level predictions** with uncertainty quantification
- **Cohort-based aggregations** for strategic planning

### 📈 Survival Analysis
- **Multiple survival models** (Kaplan-Meier, Weibull, Log-Normal)
- **Cohort-specific churn rates** by acquisition channel and plan type
- **Time-to-churn predictions** with confidence intervals
- **Retention curve modeling** for capacity planning

### 💸 Revenue Intelligence
- **Monthly revenue forecasting** per customer
- **Expansion revenue modeling** (upgrades and upsells)
- **Usage-based revenue correlation** with customer behavior
- **Seasonal adjustment** for business cyclicality

## 🛠️ Installation

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn lifelines scikit-learn

# Optional: enhanced statistical analysis
pip install scipy statsmodels
```

### Required Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
lifelines>=0.26.0
scikit-learn>=1.0.0
```

## 📝 Quick Start

### Basic LTV Forecasting

```python
from customer_ltv_forecaster import CustomerLTVForecaster

# Initialize forecaster
ltv_forecaster = CustomerLTVForecaster()

# Generate or load customer data
df = ltv_forecaster.generate_synthetic_customer_data(n_customers=10000)

# Fit survival and revenue models
survival_data = ltv_forecaster.fit_survival_models(df)
revenue_data = ltv_forecaster.fit_revenue_models(df)

# Generate probabilistic LTV forecasts
ltv_forecasts = ltv_forecaster.forecast_ltv_probabilistic(df, forecast_months=6)

# Calculate cohort-based insights
cohort_analysis = ltv_forecaster.calculate_cohort_ltv(ltv_forecasts)

print(f"Average Customer LTV: ${ltv_forecasts['ltv_median'].mean():.2f}")
print(f"12-month survival rate: {ltv_forecasts['survival_prob_12m'].mean():.1%}")
```

### Advanced Pricing Analysis

```python
# Pricing sensitivity analysis
current_prices = {'free': 0, 'basic': 29, 'pro': 99, 'enterprise': 399}
pricing_analysis = ltv_forecaster.pricing_sensitivity_analysis(ltv_forecasts, current_prices)

# Generate strategic recommendations
recommendations = ltv_forecaster.generate_pricing_recommendations(
    ltv_forecasts, pricing_analysis
)

# Create comprehensive visualizations
ltv_forecaster.plot_ltv_analysis(df, ltv_forecasts, cohort_analysis)
```

## 📁 Data Structure

### Customer Data Format
```python
df = pd.DataFrame({
    'customer_id': [1, 2, 3, ...],
    'signup_date': ['2023-01-15', '2023-01-16', ...],
    'acquisition_channel': ['organic', 'paid_search', 'social', ...],
    'plan_type': ['free', 'basic', 'pro', 'enterprise'],
    'company_size': ['1-10', '11-50', '51-200', '201-1000', '1000+'],
    'industry': ['tech', 'finance', 'healthcare', ...],
    'geography': ['US', 'EU', 'APAC', 'Other'],
    'month': pd.date_range('2023-01-01', periods=24, freq='M'),
    'revenue': [0, 29, 99, ...],  # Monthly revenue per customer
    'usage_level': [0.5, 0.8, 1.2, ...],  # Relative usage intensity
    'is_churned': [False, False, True, ...],  # Churn indicator
})
```

### Synthetic Data Generation
The solution includes realistic synthetic data that simulates:

#### Customer Acquisition Patterns
- **Organic growth**: 30% of signups, lowest churn, moderate LTV
- **Paid search**: 25% of signups, average performance
- **Social media**: 20% of signups, higher churn but viral potential
- **Referrals**: 15% of signups, highest retention and LTV
- **Email marketing**: 10% of signups, consistent performance

#### Plan Type Characteristics
- **Free (40% of users)**: $0 revenue, 10% monthly churn, high upgrade potential
- **Basic (30% of users)**: $29/month, 6% monthly churn, stable revenue
- **Pro (25% of users)**: $99/month, 4% monthly churn, expansion revenue
- **Enterprise (5% of users)**: $399/month, 2% monthly churn, highest LTV

#### Revenue Evolution Patterns
- **New customer onboarding**: 30-100% usage ramp in first 3 months
- **Expansion revenue**: 5% monthly chance of upgrades after month 3
- **Seasonal effects**: ±10% revenue variation by month
- **Usage correlation**: Revenue tied to customer engagement levels

## 🏗️ Model Architecture

### Hierarchical Modeling Approach
```
Customer LTV Forecasting
├── Survival Analysis
│   ├── Kaplan-Meier (non-parametric)
│   ├── Weibull (parametric)
│   └── Log-Normal (parametric)
├── Revenue Prediction
│   ├── Random Forest (ensemble)
│   ├── Gradient Boosting (ensemble)
│   └── Feature Engineering (lags, trends)
└── Monte Carlo Simulation
    ├── Revenue scenarios (100+ simulations)
    ├── Survival probability integration
    └── Uncertainty quantification
```

### Feature Engineering
**Customer Characteristics**: Acquisition channel, plan type, company size, industry, geography
**Behavioral Features**: Usage level, revenue history, engagement trends
**Temporal Features**: Months since signup, seasonal effects, cohort membership
**Derived Metrics**: Revenue growth, usage evolution, churn risk indicators

### LTV Calculation
```
LTV = Σ(t=1 to T) [Monthly_Revenue(t) × Survival_Probability(t) × Discount_Factor(t)]

Where:
- Monthly_Revenue(t) = Predicted revenue in month t
- Survival_Probability(t) = Probability customer survives to month t
- Discount_Factor(t) = (1 + discount_rate)^(-t)
```

## 📊 Output Formats

### LTV Forecasts
```python
ltv_forecasts = pd.DataFrame({
    'customer_id': [1, 2, 3, ...],
    'ltv_median': [850, 1200, 450, ...],     # 50th percentile LTV
    'ltv_p25': [600, 900, 300, ...],        # Conservative estimate
    'ltv_p75': [1100, 1500, 600, ...],      # Optimistic estimate
    'ltv_p10': [400, 650, 200, ...],        # Risk-adjusted planning
    'ltv_p90': [1400, 2000, 800, ...],      # Best-case scenario
    'survival_prob_12m': [0.7, 0.8, 0.6, ...], # 12-month retention
    'survival_prob_24m': [0.5, 0.7, 0.4, ...], # 24-month retention
})
```

### Cohort Analysis
```python
cohort_analysis = {
    'acquisition_channel': {
        'organic': {'ltv_median': 950, 'survival_12m': 0.75},
        'referral': {'ltv_median': 1200, 'survival_12m': 0.85},
        'paid_search': {'ltv_median': 800, 'survival_12m': 0.70},
    },
    'plan_type': {
        'enterprise': {'ltv_median': 2500, 'survival_12m': 0.90},
        'pro': {'ltv_median': 1100, 'survival_12m': 0.80},
        'basic': {'ltv_median': 450, 'survival_12m': 0.70},
    }
}
```

### Pricing Analysis
```python
pricing_scenarios = pd.DataFrame({
    'price_multiplier': [0.5, 0.75, 1.0, 1.25, 1.5],
    'demand_change_pct': [-15, -8, 0, -12, -25],
    'ltv_per_customer': [400, 600, 800, 1000, 1200],
    'total_value_index': [0.85, 0.95, 1.0, 1.1, 0.9],
})
```

## 📈 Visualization Suite

### Comprehensive Dashboard
1. **LTV Distribution**: Histogram with mean/median markers
2. **Channel Performance**: Bar charts with error bars
3. **Plan Type Analysis**: Revenue and retention comparison
4. **Survival Curves**: Kaplan-Meier retention plots
5. **Uncertainty Visualization**: Confidence interval analysis
6. **Cohort Heatmaps**: Retention patterns over time
7. **Revenue Trends**: Monthly performance by cohort
8. **LTV/CAC Analysis**: Return on acquisition investment
9. **Feature Importance**: Revenue prediction drivers

### Business Intelligence Views
- **Executive summary** with key metrics
- **Cohort performance** tracking
- **Pricing optimization** scenarios
- **Risk assessment** with uncertainty bands

## ⚙️ Customization Options

### Model Parameters
```python
# Survival model selection
survival_models = ['kaplan_meier', 'weibull', 'lognormal']

# Revenue model configuration
revenue_models = {
    'random_forest': RandomForestRegressor(n_estimators=100),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100)
}

# Forecasting parameters
forecast_config = {
    'forecast_months': 6,           # Prediction horizon
    'discount_rate': 0.01,          # Monthly discount rate (1%)
    'n_scenarios': 100,             # Monte Carlo simulations
    'confidence_levels': [0.1, 0.25, 0.5, 0.75, 0.9]
}
```

### Customer Segmentation
```python
# Custom cohort definitions
cohort_definitions = {
    'acquisition_channel': ['organic', 'paid', 'social', 'referral'],
    'plan_type': ['free', 'basic', 'pro', 'enterprise'],
    'company_size': ['1-10', '11-50', '51-200', '201-1000', '1000+'],
    'industry': ['tech', 'finance', 'healthcare', 'retail', 'education'],
    'geography': ['US', 'EU', 'APAC', 'Other']
}
```

### Price Elasticity Settings
```python
# Industry-specific elasticity
price_elasticity = {
    'free': -0.5,        # Low sensitivity
    'basic': -1.2,       # Standard elasticity
    'pro': -0.8,         # Moderate sensitivity  
    'enterprise': -0.3   # Low sensitivity
}
```

## 🎯 Business Applications

### Customer Acquisition Strategy
- **Channel optimization**: Allocate marketing budget based on LTV by source
- **CAC limits**: Set acquisition cost limits using LTV/CAC ratios
- **Quality scoring**: Prioritize high-LTV customer segments
- **Geographic expansion**: Target regions with favorable LTV profiles

### Pricing Optimization
- **Plan positioning**: Design pricing tiers based on customer value
- **Elasticity testing**: A/B test price changes on low-risk segments
- **Value-based pricing**: Align pricing with delivered customer value
- **Competitive positioning**: Price relative to customer lifetime value

### Product Strategy
- **Feature prioritization**: Develop features that increase customer LTV
- **Retention programs**: Target customers with high LTV but churn risk
- **Upselling campaigns**: Identify expansion revenue opportunities
- **Churn prevention**: Proactive intervention for high-value customers

### Financial Planning
- **Revenue forecasting**: Project revenue based on customer acquisition
- **Risk management**: Use P25 LTV for conservative financial planning
- **Investment decisions**: Evaluate marketing spend ROI using LTV projections
- **Scenario planning**: Model business performance under different assumptions

## ⚠️ Model Limitations & Assumptions

### Key Assumptions
- **Historical patterns continue**: Past customer behavior predicts future behavior
- **Market stability**: No major competitive or economic disruptions
- **Product consistency**: Core value proposition remains stable
- **Data quality**: Customer tracking and revenue attribution are accurate

### Known Limitations
- **Early customers**: Limited data for recent signups reduces accuracy
- **External shocks**: Cannot predict black swan events or market disruptions
- **Competitive dynamics**: New competitors or pricing changes not modeled
- **Product evolution**: Major feature changes may alter customer behavior

### Uncertainty Sources
- **Model uncertainty**: Different algorithms produce varying estimates
- **Parameter uncertainty**: Confidence intervals reflect estimation uncertainty
- **Scenario uncertainty**: Multiple revenue paths create forecast ranges
- **Data uncertainty**: Measurement errors and missing data impact accuracy

## 📊 Model Validation

### Backtesting Approach
```python
# Time series validation
train_end_date = '2023-06-01'
test_start_date = '2023-07-01'

# Fit models on historical data
train_data = df[df['month'] <= train_end_date]
test_data = df[df['month'] >= test_start_date]

# Validate predictions against actual outcomes
forecast_accuracy = validate_ltv_predictions(train_data, test_data)
```

### Performance Metrics
- **MAPE**: Mean Absolute Percentage Error for LTV predictions
- **Concordance Index**: Survival model discrimination ability
- **Calibration**: Predicted vs. actual survival probabilities
- **Coverage**: Percentage of actual values within confidence intervals

### Model Monitoring
- **Monthly retraining**: Update models with new data
- **Drift detection**: Monitor for changes in customer behavior
- **Performance tracking**: Compare predictions to actual outcomes
- **Alert system**: Flag significant deviations from expectations

## 🔧 Production Deployment

### Batch Processing Pipeline
```python
# Monthly LTV refresh
def monthly_ltv_update():
    """Update LTV forecasts with latest customer data"""
    latest_data = fetch_customer_data()
    updated_forecasts = ltv_forecaster.forecast_ltv_probabilistic(latest_data)
    update_customer_ltv_table(updated_forecasts)
    
# Real-time scoring
def score_new_customer(customer_attributes):
    """Score new customer LTV at signup"""
    return ltv_forecaster.predict_single_customer_ltv(customer_attributes)
```

### Integration Points
- **CRM Systems**: Customer LTV scoring and segmentation
- **Marketing Platforms**: Campaign targeting and budget allocation  
- **Finance Systems**: Revenue forecasting and planning
- **Product Analytics**: Feature impact on customer value

### Business Intelligence Integration
- **Dashboards**: Executive LTV reporting and trends
- **Alerts**: High-value customer churn warnings
- **Reporting**: Monthly cohort performance analysis
- **APIs**: Real-time LTV scoring for applications

## 🤝 Contributing

### Development Setup
```bash
git clone [repository-url]
cd customer-ltv-forecasting
pip install -r requirements.txt
python -m pytest tests/
```

### Adding New Features
1. **New survival models**: Extend model options in `fit_survival_models()`
2. **Additional cohorts**: Add segmentation dimensions
3. **Enhanced features**: Improve revenue prediction with new variables
4. **Alternative scenarios**: Expand Monte Carlo simulation scenarios

### Testing Framework
```python
# Unit tests for core functionality
pytest tests/test_survival_models.py
pytest tests/test_revenue_prediction.py

# Integration tests with sample data
pytest tests/test_ltv_pipeline.py

# Performance benchmarks
pytest tests/test_model_performance.py
```

## 📄 License

MIT License


---

**Built for customer-centric growth** 💰📈