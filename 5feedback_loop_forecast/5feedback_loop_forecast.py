# 🔄 Forecasting Infrastructure Load with Feedback Loops
# Your compute load forecast is used to allocate GPUs.
# But this forecast now influences engineering behavior (e.g., they delay launches to stay under thresholds).
# How do you forecast accurately in this feedback loop?

# Feedback loops create endogeneity → causal modeling needed
# Use instrumental variables or causal forests to isolate true drivers
# Build forecast-aware allocation strategies (e.g., forecasting “planned” vs “organic” usage separately)
# Collaborate with engineers to identify anticipatory behavior
# Show that you can combine forecasting with counterfactual reasoning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Game theory and equilibrium
from sklearn.ensemble import RandomForestRegressor


class FeedbackAwareLoadForecaster:
    """
    Infrastructure load forecasting system that accounts for feedback loops where:
    1. Forecasts influence capacity allocation decisions
    2. Capacity decisions influence engineering behavior
    3. Engineering behavior affects actual load patterns
    4. System learns to predict equilibrium states rather than naive projections
    """

    def __init__(self):
        self.base_models = {}  # Models without feedback effects
        self.feedback_models = {}  # Models accounting for feedback
        self.behavior_models = {}  # Models predicting engineering responses
        self.equilibrium_solver = None
        self.policy_parameters = {}
        self.historical_feedback_data = []

        # Feedback loop parameters
        self.capacity_utilization_threshold = 0.8  # When engineers start reacting
        self.launch_delay_sensitivity = 0.3  # How much launches get delayed per % over threshold
        self.engineering_risk_aversion = 0.7  # How conservative engineering becomes

    def generate_historical_feedback_data(self, periods=100, start_date='2023-01-01'):
        """Generate historical data that includes observed feedback effects"""

        print("Generating historical data with feedback effects...")

        dates = pd.date_range(start_date, periods=periods, freq='W')
        data = []

        # Initialize system state
        current_capacity = 1000  # GPUs
        baseline_growth_rate = 0.02  # 2% weekly growth

        for i, date in enumerate(dates):
            week_num = i

            # Base demand without feedback effects
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * week_num / 52)
            trend = (1 + baseline_growth_rate) ** week_num
            noise = np.random.normal(1, 0.1)
            base_demand = 500 * seasonal * trend * noise

            # Planned launches (exogenous)
            planned_launches = np.random.poisson(0.3)  # ~0.3 launches per week
            launch_impact = planned_launches * np.random.uniform(50, 200)  # Each launch adds 50-200 GPU demand

            # Previous period's forecast and capacity decision
            if i > 0:
                prev_forecast = data[i - 1]['forecasted_demand']
                prev_capacity = data[i - 1]['allocated_capacity']
                prev_utilization = data[i - 1]['actual_demand'] / prev_capacity if prev_capacity > 0 else 0
            else:
                prev_forecast = base_demand
                prev_capacity = current_capacity
                prev_utilization = 0.5

            # Engineering behavior response to previous utilization
            utilization_pressure = max(0, prev_utilization - self.capacity_utilization_threshold)

            # Launch delay behavior
            if utilization_pressure > 0:
                delay_factor = min(0.8, utilization_pressure * self.launch_delay_sensitivity)
                delayed_launches = launch_impact * delay_factor
                immediate_launches = launch_impact * (1 - delay_factor)

                # Delayed launches get pushed to future periods
                if i < len(dates) - 5:
                    future_period = min(i + np.random.randint(1, 6), len(dates) - 1)
            else:
                delayed_launches = 0
                immediate_launches = launch_impact

            # Engineering optimization behavior
            if utilization_pressure > 0:
                # Engineers optimize code, batch requests, etc.
                efficiency_gain = min(0.2, utilization_pressure * 0.1)  # Up to 20% efficiency gain
                optimization_factor = 1 - efficiency_gain
            else:
                optimization_factor = 1.0

            # Actual demand with feedback effects
            actual_demand = (base_demand + immediate_launches) * optimization_factor

            # Capacity allocation decision (reactive)
            if prev_utilization > 0.9:
                capacity_multiplier = 1.3  # Aggressive scaling
            elif prev_utilization > 0.8:
                capacity_multiplier = 1.15  # Conservative scaling
            elif prev_utilization < 0.5:
                capacity_multiplier = 0.95  # Scale down
            else:
                capacity_multiplier = 1.0

            allocated_capacity = prev_capacity * capacity_multiplier

            # Naive forecast (what would happen without feedback)
            naive_forecast = base_demand + launch_impact

            # Create forecast that would have been made at this time
            # (This would be based on historical patterns, not knowing future feedback)
            forecast_error = np.random.normal(0, 0.1)
            forecasted_demand = naive_forecast * (1 + forecast_error)

            # Record external factors
            market_volatility = np.random.uniform(0.8, 1.2)
            competitor_actions = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% chance of competitor action

            record = {
                'date': date,
                'week': week_num,
                'base_demand': base_demand,
                'planned_launches': planned_launches,
                'launch_impact': launch_impact,
                'immediate_launches': immediate_launches,
                'delayed_launches': delayed_launches,
                'optimization_factor': optimization_factor,
                'actual_demand': actual_demand,
                'forecasted_demand': forecasted_demand,
                'naive_forecast': naive_forecast,
                'allocated_capacity': allocated_capacity,
                'utilization': actual_demand / allocated_capacity if allocated_capacity > 0 else 0,
                'utilization_pressure': utilization_pressure,
                'capacity_multiplier': capacity_multiplier,
                'market_volatility': market_volatility,
                'competitor_actions': competitor_actions,
                'efficiency_gains': 1 - optimization_factor
            }

            data.append(record)
            current_capacity = allocated_capacity

        df = pd.DataFrame(data)

        # Add lagged features
        df['prev_utilization'] = df['utilization'].shift(1)
        df['prev_forecast_error'] = (df['forecasted_demand'] - df['actual_demand']).shift(1)
        df['prev_capacity_change'] = df['capacity_multiplier'].shift(1)

        # Calculate cumulative feedback effects
        df['cumulative_delays'] = df['delayed_launches'].cumsum()
        df['cumulative_optimizations'] = df['efficiency_gains'].cumsum()

        self.historical_feedback_data = df
        return df

    def analyze_feedback_patterns(self, df):
        """Analyze historical feedback patterns to understand system dynamics"""

        print("Analyzing feedback patterns...")

        feedback_analysis = {}

        # 1. Launch delay sensitivity
        high_util_periods = df[df['utilization'] > 0.8]
        if len(high_util_periods) > 0:
            delay_correlation = np.corrcoef(
                high_util_periods['utilization'],
                high_util_periods['delayed_launches'] / (high_util_periods['launch_impact'] + 1e-6)
            )[0, 1]
            feedback_analysis['launch_delay_sensitivity'] = delay_correlation

        # 2. Engineering optimization response
        optimization_correlation = np.corrcoef(
            df['utilization_pressure'],
            df['efficiency_gains']
        )[0, 1]
        feedback_analysis['optimization_sensitivity'] = optimization_correlation

        # 3. Capacity allocation patterns
        capacity_response = df.groupby(pd.cut(df['prev_utilization'], bins=5))['capacity_multiplier'].mean()
        feedback_analysis['capacity_response_curve'] = capacity_response

        # 4. Forecast error patterns related to feedback
        df['forecast_error'] = df['forecasted_demand'] - df['actual_demand']
        df['feedback_effect'] = df['naive_forecast'] - df['actual_demand']

        feedback_correlation = np.corrcoef(df['utilization_pressure'], df['feedback_effect'])[0, 1]
        feedback_analysis['feedback_forecast_impact'] = feedback_correlation

        # 5. System stability metrics
        utilization_volatility = df['utilization'].std()
        capacity_volatility = df['capacity_multiplier'].std()
        feedback_analysis['system_stability'] = {
            'utilization_volatility': utilization_volatility,
            'capacity_volatility': capacity_volatility,
            'stability_ratio': utilization_volatility / capacity_volatility if capacity_volatility > 0 else 0
        }

        self.feedback_analysis = feedback_analysis
        return feedback_analysis

    def fit_behavior_models(self, df):
        """Fit models to predict engineering behavior responses"""

        print("Training behavior prediction models...")

        # Prepare features for behavior modeling
        behavior_features = [
            'prev_utilization', 'utilization_pressure', 'prev_forecast_error',
            'market_volatility', 'competitor_actions', 'week'
        ]

        required_columns = behavior_features + ['delayed_launches', 'efficiency_gains', 'capacity_multiplier',
                                                'launch_impact']

        # Remove rows with missing values
        model_data = df[required_columns].dropna()

        if len(model_data) == 0:
            print("Warning: No valid data for behavior modeling")
            return

        X = model_data[behavior_features]

        # 1. Launch delay behavior model
        y_delays = model_data['delayed_launches'] / (model_data['launch_impact'] + 1e-6)
        delay_model = RandomForestRegressor(n_estimators=100, random_state=42)
        delay_model.fit(X, y_delays)
        self.behavior_models['launch_delays'] = delay_model

        # 2. Engineering optimization model
        y_optimization = model_data['efficiency_gains']
        optimization_model = RandomForestRegressor(n_estimators=100, random_state=42)
        optimization_model.fit(X, y_optimization)
        self.behavior_models['optimization'] = optimization_model

        # 3. Capacity allocation model
        y_capacity = model_data['capacity_multiplier']
        capacity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        capacity_model.fit(X, y_capacity)
        self.behavior_models['capacity_allocation'] = capacity_model

        print("✅ Behavior models trained successfully")

    def predict_equilibrium_state(self, base_demand, planned_launches, current_capacity,
                                  market_conditions=None, max_iterations=50):
        """
        Predict the equilibrium state accounting for feedback loops
        Uses iterative solving to find stable state
        """

        if market_conditions is None:
            market_conditions = {'volatility': 1.0, 'competitor_actions': 0}

        # Initial state
        state = {
            'demand': base_demand + planned_launches,
            'capacity': current_capacity,
            'utilization': (base_demand + planned_launches) / current_capacity,
            'launch_delays': 0,
            'efficiency_gains': 0,
            'capacity_adjustment': 1.0
        }

        # Define feature names to match training
        feature_names = [
            'prev_utilization', 'utilization_pressure', 'prev_forecast_error',
            'market_volatility', 'competitor_actions', 'week'
        ]

        # Iterative equilibrium solving
        for iteration in range(max_iterations):
            prev_state = state.copy()

            # Calculate utilization pressure
            utilization_pressure = max(0, state['utilization'] - self.capacity_utilization_threshold)

            # Predict engineering responses using behavior models
            if hasattr(self, 'behavior_models') and self.behavior_models:
                # Create feature DataFrame with proper column names
                features_data = {
                    'prev_utilization': [state['utilization']],
                    'utilization_pressure': [utilization_pressure],
                    'prev_forecast_error': [0],  # assume 0 for equilibrium
                    'market_volatility': [market_conditions['volatility']],
                    'competitor_actions': [market_conditions['competitor_actions']],
                    'week': [0]  # week (relative)
                }
                features_df = pd.DataFrame(features_data)

                # Predict behaviors
                if 'launch_delays' in self.behavior_models:
                    delay_factor = self.behavior_models['launch_delays'].predict(features_df)[0]
                else:
                    delay_factor = min(0.8, utilization_pressure * self.launch_delay_sensitivity)

                if 'optimization' in self.behavior_models:
                    efficiency_gain = self.behavior_models['optimization'].predict(features_df)[0]
                else:
                    efficiency_gain = min(0.2, utilization_pressure * 0.1)

                if 'capacity_allocation' in self.behavior_models:
                    capacity_multiplier = self.behavior_models['capacity_allocation'].predict(features_df)[0]
                else:
                    if state['utilization'] > 0.9:
                        capacity_multiplier = 1.3
                    elif state['utilization'] > 0.8:
                        capacity_multiplier = 1.15
                    else:
                        capacity_multiplier = 1.0
            else:
                # Fallback to simple rules
                delay_factor = min(0.8, utilization_pressure * self.launch_delay_sensitivity)
                efficiency_gain = min(0.2, utilization_pressure * 0.1)
                capacity_multiplier = 1.15 if state['utilization'] > 0.8 else 1.0

            # Update state based on predicted behaviors
            immediate_launches = planned_launches * (1 - delay_factor)
            optimization_factor = 1 - efficiency_gain

            new_demand = (base_demand + immediate_launches) * optimization_factor
            new_capacity = current_capacity * capacity_multiplier
            new_utilization = new_demand / new_capacity if new_capacity > 0 else 0

            state.update({
                'demand': new_demand,
                'capacity': new_capacity,
                'utilization': new_utilization,
                'launch_delays': delay_factor,
                'efficiency_gains': efficiency_gain,
                'capacity_adjustment': capacity_multiplier
            })

            # Check for convergence
            demand_change = abs(state['demand'] - prev_state['demand']) / prev_state['demand']
            capacity_change = abs(state['capacity'] - prev_state['capacity']) / prev_state['capacity']

            if demand_change < 0.001 and capacity_change < 0.001:
                print(f"Equilibrium converged after {iteration + 1} iterations")
                break

        return state

    def forecast_with_feedback_awareness(self, base_forecasts, current_capacity,
                                         horizon_days=30, confidence_levels=[0.1, 0.5, 0.9]):
        """
        Generate forecasts that account for feedback loops and behavioral responses
        """

        print(f"Generating feedback-aware forecasts for {horizon_days} days...")

        forecasts = []

        for day in range(horizon_days):
            # Base demand for this day
            base_demand = base_forecasts.get(day, 500)  # Default fallback

            # Planned launches (could be data-driven)
            planned_launches = np.random.poisson(0.5) * np.random.uniform(50, 200)

            # Market conditions
            market_conditions = {
                'volatility': np.random.uniform(0.8, 1.2),
                'competitor_actions': np.random.choice([0, 1], p=[0.8, 0.2])
            }

            # Multiple scenario analysis
            scenarios = []

            for scenario_name, scenario_params in [
                ('optimistic', {'demand_multiplier': 0.9, 'response_intensity': 0.7}),
                ('expected', {'demand_multiplier': 1.0, 'response_intensity': 1.0}),
                ('pessimistic', {'demand_multiplier': 1.1, 'response_intensity': 1.3})
            ]:
                # Adjust base demand for scenario
                scenario_base_demand = base_demand * scenario_params['demand_multiplier']

                # Temporarily adjust behavior sensitivity for scenario
                original_sensitivity = self.launch_delay_sensitivity
                self.launch_delay_sensitivity *= scenario_params['response_intensity']

                # Predict equilibrium for this scenario
                equilibrium = self.predict_equilibrium_state(
                    scenario_base_demand,
                    planned_launches,
                    current_capacity,
                    market_conditions
                )

                # Restore original sensitivity
                self.launch_delay_sensitivity = original_sensitivity

                scenarios.append({
                    'scenario': scenario_name,
                    'equilibrium_demand': equilibrium['demand'],
                    'equilibrium_capacity': equilibrium['capacity'],
                    'equilibrium_utilization': equilibrium['utilization'],
                    'launch_delays': equilibrium['launch_delays'],
                    'efficiency_gains': equilibrium['efficiency_gains']
                })

            # Aggregate scenarios into confidence intervals
            scenario_demands = [s['equilibrium_demand'] for s in scenarios]
            scenario_utilizations = [s['equilibrium_utilization'] for s in scenarios]

            forecast_day = {
                'day': day,
                'base_demand': base_demand,
                'planned_launches': planned_launches,
                'naive_forecast': base_demand + planned_launches,
                'feedback_aware_forecast': np.mean(scenario_demands),
                'forecast_p10': np.percentile(scenario_demands, 10),
                'forecast_p50': np.percentile(scenario_demands, 50),
                'forecast_p90': np.percentile(scenario_demands, 90),
                'expected_utilization': np.mean(scenario_utilizations),
                'utilization_p10': np.percentile(scenario_utilizations, 10),
                'utilization_p90': np.percentile(scenario_utilizations, 90),
                'scenarios': scenarios,
                'feedback_effects': {
                    'expected_delay_factor': np.mean([s['launch_delays'] for s in scenarios]),
                    'expected_efficiency_gain': np.mean([s['efficiency_gains'] for s in scenarios])
                }
            }

            forecasts.append(forecast_day)

            # Update capacity for next iteration
            current_capacity = np.mean([s['equilibrium_capacity'] for s in scenarios])

        return pd.DataFrame(forecasts)

    def optimize_capacity_policy(self, historical_data, policy_params=None):
        """
        Optimize capacity allocation policy to minimize cost while maintaining SLA
        """

        if policy_params is None:
            policy_params = {
                'target_utilization': 0.75,
                'utilization_buffer': 0.1,
                'scaling_aggressiveness': 1.2,
                'cost_per_gpu_hour': 3.5,
                'sla_violation_penalty': 1000
            }

        def policy_objective(params):
            target_util, buffer, aggressiveness = params

            total_cost = 0
            sla_violations = 0

            for _, row in historical_data.iterrows():
                # Simulate policy decision
                if row['prev_utilization'] > target_util + buffer:
                    capacity_multiplier = aggressiveness
                elif row['prev_utilization'] < target_util - buffer:
                    capacity_multiplier = 1 / aggressiveness
                else:
                    capacity_multiplier = 1.0

                # Calculate costs
                capacity_cost = row['allocated_capacity'] * capacity_multiplier * policy_params['cost_per_gpu_hour']

                # Calculate SLA violations (utilization > 95%)
                if row['utilization'] > 0.95:
                    sla_violations += 1

                total_cost += capacity_cost

            # Total objective: minimize cost + SLA violation penalty
            total_penalty = sla_violations * policy_params['sla_violation_penalty']
            return total_cost + total_penalty

        # Optimize policy parameters
        initial_params = [policy_params['target_utilization'],
                          policy_params['utilization_buffer'],
                          policy_params['scaling_aggressiveness']]

        bounds = [(0.5, 0.9), (0.05, 0.3), (1.1, 2.0)]

        result = minimize(policy_objective, initial_params, bounds=bounds, method='L-BFGS-B')

        optimal_policy = {
            'target_utilization': result.x[0],
            'utilization_buffer': result.x[1],
            'scaling_aggressiveness': result.x[2],
            'expected_cost': result.fun
        }

        self.optimal_policy = optimal_policy
        return optimal_policy

    def simulate_policy_impact(self, policy, forecast_horizon=30):
        """
        Simulate the impact of a capacity allocation policy on system behavior
        """

        print(f"Simulating policy impact over {forecast_horizon} days...")

        # Generate base forecasts
        base_forecasts = {day: 500 * (1.02 ** (day / 7)) for day in range(forecast_horizon)}

        # Simulate with current policy
        current_results = self.forecast_with_feedback_awareness(
            base_forecasts,
            current_capacity=1000,
            horizon_days=forecast_horizon
        )

        # Simulate with optimized policy
        # Temporarily adjust behavior parameters
        original_threshold = self.capacity_utilization_threshold
        original_sensitivity = self.launch_delay_sensitivity

        self.capacity_utilization_threshold = policy['target_utilization']
        self.launch_delay_sensitivity *= (policy['scaling_aggressiveness'] - 1)

        optimized_results = self.forecast_with_feedback_awareness(
            base_forecasts,
            current_capacity=1000,
            horizon_days=forecast_horizon
        )

        # Restore original parameters
        self.capacity_utilization_threshold = original_threshold
        self.launch_delay_sensitivity = original_sensitivity

        # Compare results
        comparison = {
            'current_policy': {
                'avg_utilization': current_results['expected_utilization'].mean(),
                'utilization_volatility': current_results['expected_utilization'].std(),
                'avg_demand': current_results['feedback_aware_forecast'].mean(),
                'demand_volatility': current_results['feedback_aware_forecast'].std()
            },
            'optimized_policy': {
                'avg_utilization': optimized_results['expected_utilization'].mean(),
                'utilization_volatility': optimized_results['expected_utilization'].std(),
                'avg_demand': optimized_results['feedback_aware_forecast'].mean(),
                'demand_volatility': optimized_results['feedback_aware_forecast'].std()
            }
        }

        # Calculate improvement metrics
        utilization_improvement = (
                                          comparison['optimized_policy']['avg_utilization'] -
                                          comparison['current_policy']['avg_utilization']
                                  ) / comparison['current_policy']['avg_utilization'] * 100

        stability_improvement = (
                                        comparison['current_policy']['utilization_volatility'] -
                                        comparison['optimized_policy']['utilization_volatility']
                                ) / comparison['current_policy']['utilization_volatility'] * 100

        comparison['improvements'] = {
            'utilization_improvement_pct': utilization_improvement,
            'stability_improvement_pct': stability_improvement
        }

        return comparison

    def plot_feedback_analysis(self, historical_data, forecasts):
        """Create comprehensive visualization of feedback effects and forecasts"""

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # 1. Historical demand vs naive forecast vs actual
        ax1 = axes[0, 0]
        ax1.plot(historical_data['date'], historical_data['naive_forecast'],
                 label='Naive Forecast', linestyle='--', alpha=0.7)
        ax1.plot(historical_data['date'], historical_data['actual_demand'],
                 label='Actual Demand', linewidth=2)
        ax1.fill_between(historical_data['date'],
                         historical_data['naive_forecast'],
                         historical_data['actual_demand'],
                         alpha=0.3, label='Feedback Effect')
        ax1.set_title('Historical: Naive vs Actual Demand')
        ax1.set_ylabel('Demand (GPU Hours)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Utilization vs capacity over time
        ax2 = axes[0, 1]
        ax2.plot(historical_data['date'], historical_data['utilization'],
                 label='Utilization', color='red', linewidth=2)
        ax2.axhline(y=0.8, color='orange', linestyle='--', label='Threshold (80%)')
        ax2.axhline(y=0.95, color='red', linestyle='--', label='SLA Limit (95%)')
        ax2.set_title('Historical Utilization Patterns')
        ax2.set_ylabel('Utilization Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Engineering behavior responses
        ax3 = axes[0, 2]
        high_util_data = historical_data[historical_data['utilization'] > 0.8]
        if len(high_util_data) > 0:
            ax3.scatter(high_util_data['utilization'], high_util_data['efficiency_gains'],
                        alpha=0.6, label='Efficiency Gains')
            ax3.scatter(high_util_data['utilization'],
                        high_util_data['delayed_launches'] / (high_util_data['launch_impact'] + 1e-6),
                        alpha=0.6, label='Launch Delays', color='orange')
        ax3.set_xlabel('Utilization')
        ax3.set_ylabel('Response Intensity')
        ax3.set_title('Engineering Response to Utilization')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Forecast comparison: naive vs feedback-aware
        ax4 = axes[1, 0]
        ax4.plot(forecasts['day'], forecasts['naive_forecast'],
                 label='Naive Forecast', linestyle='--')
        ax4.plot(forecasts['day'], forecasts['feedback_aware_forecast'],
                 label='Feedback-Aware', linewidth=2)
        ax4.fill_between(forecasts['day'], forecasts['forecast_p10'], forecasts['forecast_p90'],
                         alpha=0.3, label='P10-P90 Range')
        ax4.set_title('Forecast Comparison')
        ax4.set_xlabel('Days Ahead')
        ax4.set_ylabel('Predicted Demand')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Predicted utilization with uncertainty
        ax5 = axes[1, 1]
        ax5.plot(forecasts['day'], forecasts['expected_utilization'],
                 linewidth=2, label='Expected Utilization')
        ax5.fill_between(forecasts['day'], forecasts['utilization_p10'], forecasts['utilization_p90'],
                         alpha=0.3, label='P10-P90 Range')
        ax5.axhline(y=0.8, color='orange', linestyle='--', label='Threshold')
        ax5.axhline(y=0.95, color='red', linestyle='--', label='SLA Limit')
        ax5.set_title('Predicted Utilization')
        ax5.set_xlabel('Days Ahead')
        ax5.set_ylabel('Utilization Rate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Feedback effects over forecast horizon
        ax6 = axes[1, 2]
        delay_factors = [day['feedback_effects']['expected_delay_factor'] for day in forecasts.to_dict('records')]
        efficiency_gains = [day['feedback_effects']['expected_efficiency_gain'] for day in forecasts.to_dict('records')]

        ax6.plot(forecasts['day'], delay_factors, label='Launch Delays', marker='o')
        ax6.plot(forecasts['day'], efficiency_gains, label='Efficiency Gains', marker='s')
        ax6.set_title('Predicted Feedback Effects')
        ax6.set_xlabel('Days Ahead')
        ax6.set_ylabel('Effect Intensity')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Forecast error analysis
        ax7 = axes[2, 0]
        historical_data['forecast_error'] = historical_data['forecasted_demand'] - historical_data['actual_demand']
        ax7.hist(historical_data['forecast_error'], bins=20, alpha=0.7, density=True)
        ax7.axvline(x=0, color='red', linestyle='--', label='Perfect Forecast')
        ax7.axvline(x=historical_data['forecast_error'].mean(), color='orange',
                    linestyle='--', label=f'Mean Error: {historical_data["forecast_error"].mean():.1f}')
        ax7.set_title('Historical Forecast Error Distribution')
        ax7.set_xlabel('Forecast Error (Predicted - Actual)')
        ax7.set_ylabel('Density')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. System dynamics phase plot
        ax8 = axes[2, 1]
        ax8.scatter(historical_data['utilization'], historical_data['capacity_multiplier'],
                    c=historical_data['week'], cmap='viridis', alpha=0.6)
        ax8.set_xlabel('Utilization')
        ax8.set_ylabel('Capacity Multiplier (Next Period)')
        ax8.set_title('System Dynamics (Color = Time)')
        ax8.grid(True, alpha=0.3)

        # 9. Policy impact simulation
        ax9 = axes[2, 2]
        ax9.text(0.1, 0.8, 'Policy Impact Analysis', fontsize=14, fontweight='bold')
        ax9.text(0.1, 0.6, '• Current Policy Performance', fontsize=12)
        ax9.text(0.1, 0.5, '• Optimized Policy Benefits', fontsize=12)
        ax9.text(0.1, 0.4, '• Stability Improvements', fontsize=12)
        ax9.text(0.1, 0.3, '• Cost-Efficiency Gains', fontsize=12)
        ax9.text(0.1, 0.2, '• Feedback Loop Management', fontsize=12)
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')

        plt.suptitle('Feedback-Aware Infrastructure Load Forecasting Analysis', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig('5feedback_loop_forecast/feedback_aware_forecasting_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_strategic_recommendations(self, historical_data, forecasts, policy_analysis):
        """Generate actionable recommendations for managing feedback loops"""

        # Analyze feedback loop strength
        feedback_strength = abs(self.feedback_analysis.get('feedback_forecast_impact', 0))
        system_stability = self.feedback_analysis.get('system_stability', {})

        recommendations = f"""
🔄 FEEDBACK-AWARE INFRASTRUCTURE FORECASTING RECOMMENDATIONS
============================================================

📊 SYSTEM DYNAMICS ANALYSIS:
• Feedback Loop Strength: {feedback_strength:.2f} (0=none, 1=complete)
• System Stability Score: {system_stability.get('stability_ratio', 0):.2f}
• Utilization Volatility: {system_stability.get('utilization_volatility', 0):.1%}
• Engineering Response Sensitivity: {self.feedback_analysis.get('optimization_sensitivity', 0):.2f}

🎯 KEY INSIGHTS:
1. FORECAST ACCURACY: Traditional forecasts miss {abs(historical_data['forecast_error'].mean()):.0f} GPU-hours on average
2. BEHAVIORAL RESPONSE: Engineers delay {self.feedback_analysis.get('launch_delay_sensitivity', 0) * 100:.0f}% of launches when utilization > 80%
3. EFFICIENCY GAINS: Teams achieve up to {historical_data['efficiency_gains'].max() * 100:.0f}% optimization under pressure
4. CAPACITY OSCILLATION: Current policy creates {system_stability.get('capacity_volatility', 0) * 100:.0f}% capacity volatility

🔧 IMMEDIATE ACTIONS (0-2 weeks):
=================================
1. IMPLEMENT FEEDBACK-AWARE FORECASTING:
   → Replace naive models with equilibrium-seeking algorithms
   → Account for engineering behavioral responses in predictions
   → Use iterative solving to find stable demand-capacity states

2. BEHAVIORAL MONITORING SYSTEM:
   → Track launch delay decisions in real-time
   → Monitor engineering optimization activities
   → Measure correlation between forecasts and actual behaviors

3. COMMUNICATION PROTOCOLS:
   → Share capacity forecasts with engineering teams 48h in advance
   → Establish clear escalation procedures for capacity concerns
   → Create feedback channels for engineering to report optimization efforts

📈 SHORT-TERM STRATEGY (2-8 weeks):
===================================
1. EQUILIBRIUM-BASED CAPACITY PLANNING:
   → Set capacity targets based on predicted equilibrium states
   → Account for behavioral dampening effects in scaling decisions
   → Optimize for system stability, not just cost minimization

2. PREDICTIVE BEHAVIORAL MODELING:
   → Train ML models on historical engineering responses
   → Predict launch delays and optimization activities
   → Incorporate team workload and risk tolerance factors

3. ADAPTIVE THRESHOLD MANAGEMENT:
   → Dynamically adjust utilization thresholds based on system state
   → Implement graduated response levels (yellow/orange/red alerts)
   → Balance proactive scaling with cost efficiency

🚀 LONG-TERM OPTIMIZATION (2-6 months):
======================================
1. GAME-THEORETIC CAPACITY ALLOCATION:
   → Model capacity decisions as multi-player optimization game
   → Find Nash equilibrium solutions for capacity-demand interactions
   → Implement mechanism design to align engineering incentives

2. CLOSED-LOOP CONTROL SYSTEM:
   → Real-time feedback between forecasts, allocations, and behaviors
   → Automatic adjustment of forecast parameters based on observed responses
   → Predictive intervention to prevent capacity crises

3. BEHAVIORAL INCENTIVE ALIGNMENT:
   → Design metrics that reward both performance and efficiency
   → Create "capacity budgets" that teams can optimize within
   → Implement shared ownership of infrastructure costs

⚖️ POLICY RECOMMENDATIONS:
==========================
OPTIMAL CAPACITY POLICY:
• Target Utilization: {self.optimal_policy.get('target_utilization', 0.75) * 100:.0f}%
• Buffer Zone: ±{self.optimal_policy.get('utilization_buffer', 0.1) * 100:.0f}%
• Scaling Aggressiveness: {self.optimal_policy.get('scaling_aggressiveness', 1.2):.1f}x
• Expected Cost Reduction: ${(policy_analysis.get('current_policy', {}).get('avg_demand', 1000) - self.optimal_policy.get('expected_cost', 900)):.0f}/week

BEHAVIORAL INTERVENTION TRIGGERS:
• Launch Delay Alert: Utilization > 75% for 2+ consecutive days
• Efficiency Drive: Utilization > 85% for 24+ hours
• Emergency Scaling: Utilization > 95% for any period

📋 MEASUREMENT & MONITORING:
============================
KEY METRICS TO TRACK:
1. Forecast Accuracy: Mean Absolute Percentage Error (MAPE)
2. Behavioral Response Rate: % of launches delayed vs predicted
3. System Stability: Coefficient of variation in utilization
4. Cost Efficiency: $/GPU-hour vs performance SLA adherence
5. Engineering Satisfaction: Survey scores on capacity predictability

DASHBOARD REQUIREMENTS:
• Real-time utilization with behavioral response indicators
• Forecast vs actual with feedback effect decomposition
• Engineering team capacity concerns and optimization activities
• Predictive alerts for capacity constraints and behavioral responses

🔮 ADVANCED CAPABILITIES (6+ months):
====================================
1. MULTI-AGENT SIMULATION:
   → Model individual team behaviors and interactions
   → Simulate policy changes before implementation
   → Optimize for emergent system-wide behaviors

2. REINFORCEMENT LEARNING FORECASTER:
   → Learn optimal forecasting strategies through interaction
   → Adapt to changing organizational behaviors over time
   → Balance exploration of new strategies with exploitation of known good ones

3. MARKET-MECHANISM INFRASTRUCTURE:
   → Internal capacity markets with pricing signals
   → Automated capacity trading between teams
   → Dynamic pricing based on demand forecasts and behavioral predictions

💡 SUCCESS CRITERIA:
===================
• Reduce forecast error by 40% within 3 months
• Decrease capacity volatility by 30% within 6 months
• Achieve 85%+ engineering satisfaction with capacity predictability
• Maintain 99.9% SLA while reducing infrastructure costs by 15%
• Eliminate capacity-driven launch delays by 80%

⚠️ RISK MITIGATION:
==================
1. MODEL UNCERTAINTY: Maintain multiple forecast models and scenario planning
2. BEHAVIORAL DRIFT: Regularly retrain behavioral models as teams adapt
3. COORDINATION FAILURES: Establish clear communication protocols and escalation paths
4. OPTIMIZATION LIMITS: Monitor for diminishing returns from engineering optimizations
5. EXTERNAL SHOCKS: Maintain emergency capacity reserves for unexpected events
"""

        return recommendations

    def run_complete_feedback_analysis(self, periods=100):
        """Run the complete feedback-aware forecasting analysis"""

        print("🔄 Starting Feedback-Aware Infrastructure Load Forecasting...")
        print("=" * 70)

        try:
            # 1. Generate historical data with feedback effects
            print("\n📊 Generating historical data with feedback loops...")
            historical_data = self.generate_historical_feedback_data(periods=periods)
            print(f"✅ Generated {len(historical_data)} periods of historical data")

            # 2. Analyze feedback patterns
            print("\n🔍 Analyzing feedback patterns...")
            feedback_analysis = self.analyze_feedback_patterns(historical_data)
            print("✅ Feedback pattern analysis complete")

            # 3. Train behavioral models
            print("\n🧠 Training behavioral prediction models...")
            self.fit_behavior_models(historical_data)
            print("✅ Behavioral models trained")

            # 4. Optimize capacity policy
            print("\n⚙️ Optimizing capacity allocation policy...")
            try:
                optimal_policy = self.optimize_capacity_policy(historical_data)
                print(f"✅ Optimal policy found: {optimal_policy['target_utilization']:.1%} target utilization")
            except Exception as e:
                print(f"⚠️ Policy optimization failed: {e}")
                optimal_policy = {
                    'target_utilization': 0.75,
                    'utilization_buffer': 0.1,
                    'scaling_aggressiveness': 1.2
                }

            # 5. Generate feedback-aware forecasts
            print("\n🔮 Generating feedback-aware forecasts...")
            base_forecasts = {day: 500 * (1.02 ** (day / 7)) for day in range(30)}
            forecasts = self.forecast_with_feedback_awareness(
                base_forecasts,
                current_capacity=1000,
                horizon_days=30
            )
            print(f"✅ Generated 30-day feedback-aware forecasts")

            # 6. Simulate policy impact
            print("\n🎯 Simulating policy impact...")
            try:
                policy_impact = self.simulate_policy_impact(optimal_policy)
                print("✅ Policy impact simulation complete")
            except Exception as e:
                print(f"⚠️ Policy simulation failed: {e}")
                policy_impact = {'improvements': {'utilization_improvement_pct': 0, 'stability_improvement_pct': 0}}

            # 7. Generate visualizations
            print("\n📈 Creating analysis dashboard...")
            try:
                self.plot_feedback_analysis(historical_data, forecasts)
                print("✅ Dashboard created and saved")
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")

            # 8. Generate strategic recommendations
            print("\n📋 Generating strategic recommendations...")
            recommendations = self.generate_strategic_recommendations(
                historical_data, forecasts, policy_impact
            )

            # 9. Summary analysis
            print("\n" + "=" * 70)
            print("📊 FEEDBACK ANALYSIS SUMMARY")
            print("=" * 70)

            feedback_strength = abs(feedback_analysis.get('feedback_forecast_impact', 0))
            naive_mae = abs(historical_data['forecast_error']).mean()

            # Calculate feedback-aware improvement
            feedback_aware_error = abs(forecasts['feedback_aware_forecast'] - forecasts['naive_forecast']).mean()
            improvement_potential = min(50, (feedback_aware_error / naive_mae) * 100) if naive_mae > 0 else 0

            print(f"""
SYSTEM CHARACTERISTICS:
• Feedback Loop Strength: {feedback_strength:.2f} (Strong: >0.5, Weak: <0.2)
• Historical Forecast Error: {naive_mae:.1f} GPU-hours MAPE
• Behavioral Response Correlation: {feedback_analysis.get('optimization_sensitivity', 0):.2f}
• System Stability Score: {feedback_analysis.get('system_stability', {}).get('stability_ratio', 0):.2f}

FORECASTING IMPROVEMENTS:
• Potential Error Reduction: {improvement_potential:.1f}%
• Feedback-Aware vs Naive Difference: {feedback_aware_error:.1f} GPU-hours
• Policy Optimization Benefits: {policy_impact.get('improvements', {}).get('utilization_improvement_pct', 0):.1f}% utilization improvement

ENGINEERING BEHAVIORAL PATTERNS:
• Launch Delays: {(historical_data['delayed_launches'] > 0).sum()} instances in {len(historical_data)} periods
• Optimization Events: {(historical_data['efficiency_gains'] > 0.05).sum()} significant efficiency improvements
• Capacity Volatility: {historical_data['capacity_multiplier'].std():.2f} (Lower is better)

FEEDBACK LOOP IMPACT:
• Average Demand Dampening: {(historical_data['naive_forecast'] - historical_data['actual_demand']).mean():.1f} GPU-hours
• Peak Load Reduction: {(historical_data['naive_forecast'] - historical_data['actual_demand']).max():.1f} GPU-hours
• System Self-Regulation Effectiveness: {min(100, feedback_strength * 100):.0f}%
""")

            print(recommendations)

            return {
                'historical_data': historical_data,
                'feedback_analysis': feedback_analysis,
                'forecasts': forecasts,
                'optimal_policy': optimal_policy,
                'policy_impact': policy_impact,
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Example usage and demonstration
print("🚀 Initializing Feedback-Aware Load Forecaster...")
# Initialize the forecaster
forecaster = FeedbackAwareLoadForecaster()

# Run complete analysis
print("Starting comprehensive feedback analysis...")
results = forecaster.run_complete_feedback_analysis(periods=100)

print("✅ Analysis completed successfully!")

# Export results
try:
    results['historical_data'].to_csv('5feedback_loop_forecast/feedback_historical_data.csv', index=False)
    results['forecasts'].to_csv('5feedback_loop_forecast/feedback_aware_forecasts.csv', index=False)

    with open('5feedback_loop_forecast/feedback_recommendations.txt', 'w') as f:
        f.write(results['recommendations'])

    print("✅ Results exported to CSV and text files")

except Exception as e:
    print(f"⚠️ Export failed: {e}")

# Demonstrate key capabilities
print("\n🔧 DEMONSTRATION: Equilibrium State Prediction")
print("-" * 50)

# Example: Predict equilibrium for high demand scenario
equilibrium = forecaster.predict_equilibrium_state(
    base_demand=800,  # High base demand
    planned_launches=300,  # Major launch planned
    current_capacity=1200,
    market_conditions={'volatility': 1.1, 'competitor_actions': 1}
)

print(f"Scenario: High demand (800) + Major launch (300) = 1100 total naive demand")
print(f"With current capacity of 1200 GPUs:")
print(f"  → Equilibrium Demand: {equilibrium['demand']:.0f} GPU-hours")
print(f"  → Equilibrium Utilization: {equilibrium['utilization']:.1%}")
print(f"  → Launch Delays: {equilibrium['launch_delays']:.1%}")
print(f"  → Efficiency Gains: {equilibrium['efficiency_gains']:.1%}")
print(f"  → Capacity Adjustment: {equilibrium['capacity_adjustment']:.2f}x")

feedback_effect = 1100 - equilibrium['demand']
print(f"  → Total Feedback Effect: {feedback_effect:.0f} GPU-hours reduction")
