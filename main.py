import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Explainable Forecasting", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Helvetica Neue', sans-serif; }
    
    .confidence-high {
        background: linear-gradient(135deg, #1e3a20 0%, #2d5a2f 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #3a3520 0%, #5a4f2f 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FFA726;
        margin: 10px 0;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #3a2020 0%, #5a2f2f 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #EF5350;
        margin: 10px 0;
    }
    
    .precedent-card {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #3b82f6;
        margin: 10px 0;
    }
    
    .risk-alert {
        background-color: #2d1b1b;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
        margin: 10px 0;
    }
    
    .metric-big {
        font-size: 2.5em;
        font-weight: bold;
        color: #00CC96;
    }
    
    .stMetric {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION & LOADING
# =============================================================================

@st.cache_data
def generate_synthetic_data():
    """Enhanced synthetic data with realistic patterns"""
    np.random.seed(42)
    n_days = 730
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Features
    price = 100 + np.random.normal(0, 5, n_days).cumsum() * 0.1
    price = np.clip(price, 85, 115)
    
    marketing = 1000 + np.random.normal(0, 200, n_days).cumsum() * 0.5
    marketing = np.clip(marketing, 500, 3000)
    
    competitor_promo = np.random.choice([0, 1], size=n_days, p=[0.92, 0.08])
    
    # Complex realistic patterns
    t = np.arange(n_days)
    trend = 0.08 * t
    weekly = 25 * np.sin(2 * np.pi * t / 7)
    monthly = 15 * np.sin(2 * np.pi * t / 30)
    annual = 40 * np.sin(2 * np.pi * t / 365)
    
    # Non-linear interactions
    base = 500
    sales = (base + trend + weekly + monthly + annual 
             - 2.8 * (price - 100) 
             + 0.15 * np.sqrt(marketing)
             - 45 * competitor_promo
             + 0.01 * (price - 100) * marketing / 1000  # Interaction
             + np.random.normal(0, 5, n_days))
    
    return pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'Price': price,
        'Marketing': marketing,
        'Competitor_Promo': competitor_promo
    })

def load_user_data(file):
    """Load and validate user data"""
    if file is None:
        return generate_synthetic_data()
    
    try:
        df = pd.read_csv(file)
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col])
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return generate_synthetic_data()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, date_col, target_col, driver_cols):
    """Create time series features"""
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    
    # Time features
    df['DayOfWeek'] = df[date_col].dt.dayofweek
    df['Month'] = df[date_col].dt.month
    df['Quarter'] = df[date_col].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['DayOfMonth'] = df[date_col].dt.day
    
    # Cyclical encoding
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Lags and rolling features
    for lag in [1, 7, 14, 28]:
        df[f'Lag_{lag}'] = df[target_col].shift(lag)
    
    for window in [7, 14, 28]:
        df[f'RollingMean_{window}'] = df[target_col].rolling(window).mean().shift(1)
        df[f'RollingStd_{window}'] = df[target_col].rolling(window).std().shift(1)
    
    # Momentum
    df['Momentum_7'] = df[target_col].diff(7)
    
    df = df.dropna().reset_index(drop=True)
    
    engineered = ['DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'DayOfMonth',
                  'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos',
                  'Lag_1', 'Lag_7', 'Lag_14', 'Lag_28',
                  'RollingMean_7', 'RollingMean_14', 'RollingMean_28',
                  'RollingStd_7', 'RollingStd_14', 'RollingStd_28',
                  'Momentum_7']
    
    return df, driver_cols + engineered

# =============================================================================
# ENSEMBLE MODELING WITH CONFIDENCE INTERVALS
# =============================================================================

class ConfidenceEnsemble:
    """
    Ensemble that provides not just predictions but confidence intervals
    based on model agreement and historical errors
    """
    
    def __init__(self):
        self.models = {
            'GradientBoost': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }
        self.scaler = StandardScaler()
        self.feature_names = None
        self.historical_errors = None
        
    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        
        # Train all models
        predictions = []
        for name, model in self.models.items():
            if name == 'Ridge':
                model.fit(X_scaled, y)
                pred = model.predict(X_scaled)
            else:
                model.fit(X, y)
                pred = model.predict(X)
            predictions.append(pred)
        
        # Calculate historical errors for each model
        predictions = np.array(predictions)
        ensemble_pred = predictions.mean(axis=0)
        self.historical_errors = y - ensemble_pred
        
        return self
    
    def predict_with_confidence(self, X):
        """Returns mean prediction, std, and confidence intervals"""
        predictions = []
        
        for name, model in self.models.items():
            if name == 'Ridge':
                X_scaled = self.scaler.transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Ensemble mean
        mean_pred = predictions.mean(axis=0)
        
        # Model disagreement (variance across models)
        model_std = predictions.std(axis=0)
        
        # Historical error distribution
        error_std = np.std(self.historical_errors)
        
        # Combined confidence interval
        # 68% confidence (1 sigma)
        ci_68_lower = mean_pred - (model_std + error_std)
        ci_68_upper = mean_pred + (model_std + error_std)
        
        # 95% confidence (2 sigma)
        ci_95_lower = mean_pred - 2 * (model_std + error_std)
        ci_95_upper = mean_pred + 2 * (model_std + error_std)
        
        return {
            'mean': mean_pred,
            'model_std': model_std,
            'error_std': error_std,
            'ci_68': (ci_68_lower, ci_68_upper),
            'ci_95': (ci_95_lower, ci_95_upper),
            'model_predictions': predictions
        }
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        # Average importance across tree models
        gb_importance = self.models['GradientBoost'].feature_importances_
        rf_importance = self.models['RandomForest'].feature_importances_
        
        avg_importance = (gb_importance + rf_importance) / 2
        
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': avg_importance
        }).sort_values('Importance', ascending=False)

# =============================================================================
# HISTORICAL PRECEDENT FINDER
# =============================================================================

def find_similar_periods(current_pattern, historical_data, date_col, target_col, n_matches=3):
    """
    Find historical periods that looked similar to current conditions
    Returns what happened AFTER those similar periods
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    window_size = len(current_pattern)
    historical_windows = []
    start_indices = []
    
    # Create sliding windows
    for i in range(len(historical_data) - window_size - 30):  # -30 to see what happened after
        window = historical_data.iloc[i:i+window_size][target_col].values
        historical_windows.append(window)
        start_indices.append(i)
    
    if not historical_windows:
        return []
    
    historical_windows = np.array(historical_windows)
    current = current_pattern[target_col].values.reshape(1, -1)
    
    # Calculate similarity
    similarities = cosine_similarity(current, historical_windows)[0]
    
    # Get top N matches
    top_indices = similarities.argsort()[-n_matches:][::-1]
    
    matches = []
    for idx in top_indices:
        start_idx = start_indices[idx]
        match_start = historical_data.iloc[start_idx][date_col]
        match_window = historical_data.iloc[start_idx:start_idx+window_size]
        after_window = historical_data.iloc[start_idx+window_size:start_idx+window_size+30]
        
        if len(after_window) > 0:
            matches.append({
                'start_date': match_start,
                'similarity': similarities[idx],
                'before': match_window,
                'after': after_window,
                'avg_before': match_window[target_col].mean(),
                'avg_after': after_window[target_col].mean(),
                'change_pct': ((after_window[target_col].mean() - match_window[target_col].mean()) 
                              / match_window[target_col].mean() * 100)
            })
    
    return matches

# =============================================================================
# RISK SCENARIO GENERATOR
# =============================================================================

def generate_risk_scenarios(base_forecast, drivers_current, driver_cols):
    """
    Generate scenarios that answer: "What would have to change for this to be wrong?"
    """
    scenarios = []
    
    for driver in driver_cols:
        # Downside scenario
        scenarios.append({
            'name': f'{driver} Drops 20%',
            'type': 'downside',
            'driver': driver,
            'change': -20,
            'likelihood': 'Medium' if 'price' in driver.lower() else 'Low'
        })
        
        # Upside scenario
        scenarios.append({
            'name': f'{driver} Increases 20%',
            'type': 'upside',
            'driver': driver,
            'change': 20,
            'likelihood': 'Medium'
        })
    
    # Extreme scenario
    scenarios.append({
        'name': 'Market Shock (All Drivers Deteriorate)',
        'type': 'extreme_downside',
        'driver': 'all',
        'change': -30,
        'likelihood': 'Very Low'
    })
    
    return scenarios

# =============================================================================
# MAIN APP
# =============================================================================

# Sidebar
with st.sidebar:
    st.title("Setup")
    
    uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
    raw_df = load_user_data(uploaded)
    
    st.success(f"✓ {len(raw_df)} rows loaded")
    
    # Column selection
    all_cols = list(raw_df.columns)
    date_col = st.selectbox("Date Column", all_cols, 
                            index=next((i for i, c in enumerate(all_cols) if 'date' in c.lower()), 0))
    
    try:
        raw_df[date_col] = pd.to_datetime(raw_df[date_col])
    except:
        st.error("Invalid date column")
        st.stop()
    
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    target_col = st.selectbox("Forecast Variable", numeric_cols)
    driver_cols = st.multiselect("Influential Factors (Drivers)", 
                                 [c for c in numeric_cols if c != target_col],
                                 default=[c for c in numeric_cols if c != target_col][:2])
    
    if not driver_cols:
        st.warning("Select at least one driver")
        st.stop()
    
    st.markdown("---")
    st.markdown("### Scenario Planning")
    scenario_changes = {}
    for driver in driver_cols:
        scenario_changes[driver] = st.slider(
            f"{driver}", -50, 50, 0, 
            format="%d%%", key=f"scen_{driver}"
        )

# Train model
with st.spinner("Training models..."):
    processed_df, feature_cols = engineer_features(raw_df, date_col, target_col, driver_cols)
    
    # Split
    train_size = len(processed_df) - 60
    train_df = processed_df.iloc[:train_size]
    test_df = processed_df.iloc[train_size:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Train ensemble
    ensemble = ConfidenceEnsemble()
    ensemble.fit(X_train, y_train, feature_cols)
    
    # Evaluate
    test_preds = ensemble.predict_with_confidence(X_test)
    mape = mean_absolute_percentage_error(y_test, test_preds['mean'])
    mae = mean_absolute_error(y_test, test_preds['mean'])

# Header
st.title(f"Forecast: {target_col}")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", f"{(1-mape)*100:.1f}%", 
              help="How often the model was right in testing")
with col2:
    confidence_score = 100 - (test_preds['model_std'].mean() / test_preds['mean'].mean() * 100)
    st.metric("Confidence Score", f"{confidence_score:.0f}/100",
              help="Based on model agreement")
with col3:
    st.metric("Avg Error", f"±{mae:.0f}",
              help="Typical prediction error in actual units")
with col4:
    recent_trend = (processed_df[target_col].iloc[-7:].mean() - 
                   processed_df[target_col].iloc[-14:-7].mean()) / processed_df[target_col].iloc[-14:-7].mean() * 100
    st.metric("Recent Trend", f"{recent_trend:+.1f}%")

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast",
    "🔍 Historical Precedent",
    "⚠️ Risk Scenarios",
    "🎯 Forecast Drivers"
])

# TAB 1: Forecast
with tab1:
    
    # Generate future forecast
    last_date = processed_df[date_col].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=30, freq='D')
    
    # Simplified future feature generation
    recent = processed_df.tail(60)
    future_df = pd.DataFrame({date_col: future_dates})
    
    # Apply scenarios to drivers
    for driver in driver_cols:
        base_val = recent[driver].mean()
        pct = scenario_changes.get(driver, 0)
        future_df[driver] = base_val * (1 + pct/100)
    
    # Calendar features
    future_df['DayOfWeek'] = future_df[date_col].dt.dayofweek
    future_df['Month'] = future_df[date_col].dt.month
    future_df['Quarter'] = future_df[date_col].dt.quarter
    future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
    future_df['DayOfMonth'] = future_df[date_col].dt.day
    future_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
    future_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
    future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
    future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
    
    # Lags (use recent history)
    for lag in [1, 7, 14, 28]:
        future_df[f'Lag_{lag}'] = recent[target_col].iloc[-lag:].mean()
    
    for window in [7, 14, 28]:
        future_df[f'RollingMean_{window}'] = recent[target_col].tail(window).mean()
        future_df[f'RollingStd_{window}'] = recent[target_col].tail(window).std()
    
    future_df['Momentum_7'] = recent[target_col].diff().tail(7).mean()
    
    # Predict
    X_future = future_df[feature_cols]
    future_preds = ensemble.predict_with_confidence(X_future)
    
    # Visualization
    fig = go.Figure()
    
    # Historical
    hist = processed_df.tail(90)
    fig.add_trace(go.Scatter(
        x=hist[date_col], y=hist[target_col],
        name='Historical', line=dict(color='gray', width=2),
        hovertemplate='%{y:.0f}<extra></extra>'
    ))
    
    # Forecast mean
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['mean'],
        name='Most Likely', line=dict(color='#00CC96', width=4),
        hovertemplate='%{y:.0f}<extra></extra>'
    ))
    
    # 68% confidence band
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_68'][1],
        fill=None, mode='lines', line_color='rgba(0,204,150,0)',
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_68'][0],
        fill='tonexty', mode='lines', line_color='rgba(0,204,150,0)',
        name='68% Confidence', fillcolor='rgba(0,204,150,0.3)',
        hovertemplate='Range: %{y:.0f}<extra></extra>'
    ))
    
    # 95% confidence band
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_95'][1],
        fill=None, mode='lines', line_color='rgba(0,204,150,0)',
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_95'][0],
        fill='tonexty', mode='lines', line_color='rgba(0,204,150,0)',
        name='95% Confidence', fillcolor='rgba(0,204,150,0.15)',
        hovertemplate='Range: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Interpretation
    avg_forecast = future_preds['mean'].mean()
    avg_historical = recent[target_col].tail(30).mean()
    pct_change = (avg_forecast - avg_historical) / avg_historical * 100
    
    ci_width = (future_preds['ci_68'][1].mean() - future_preds['ci_68'][0].mean())
    confidence_level = "HIGH" if ci_width < avg_forecast * 0.2 else "MEDIUM" if ci_width < avg_forecast * 0.4 else "LOW"
    
    if confidence_level == "HIGH":
        emoji = "✅"
    elif confidence_level == "MEDIUM":
        emoji = "⚠️"
    else:
        emoji = "🔴"
    
    st.markdown(f"{emoji} Confidence: {confidence_level}")
    st.markdown(f"**Most Likely Outcome:** {target_col} will average **{avg_forecast:.0f}** over the next 30 days ({pct_change:+.1f}% vs recent history)")
    st.markdown(f"**What this means**:")
    st.markdown(f"* There's a 68% chance the actual result will be within ±{ci_width/2:.0f} of this prediction")
    st.markdown(f"* There's a 95% chance it will be within ±{(future_preds['ci_95'][1].mean() - future_preds['ci_95'][0].mean())/2:.0f}")

    # Scenario impact
    if any(v != 0 for v in scenario_changes.values()):
        st.markdown("### 🎲 Scenario Impact")
        active_changes = {k: v for k, v in scenario_changes.items() if v != 0}
        for driver, change in active_changes.items():
            direction = "↑" if change > 0 else "↓"
            st.markdown(f"- **{driver}** {direction} {abs(change)}%")

# TAB 2: Historical Precedent
with tab2:
    st.markdown("### 🔍 \"When did this happen before, and what happened next?\"")
    
    current_window = processed_df.tail(30)
    matches = find_similar_periods(current_window, processed_df.head(train_size), 
                                   date_col, target_col, n_matches=3)
    
    if matches:
        for i, match in enumerate(matches):
            st.markdown(f"""
            <div class="precedent-card">
                <h4>📅 Match #{i+1}: {match['start_date'].strftime('%B %Y')} ({match['similarity']*100:.0f}% similarity)</h4>
                <p><strong>What happened then:</strong> {target_col} averaged {match['avg_before']:.0f} during the period, 
                then changed to {match['avg_after']:.0f} in the following 30 days 
                (<strong>{match['change_pct']:+.1f}%</strong>)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Side by side comparison
            col_a, col_b = st.columns(2)
            with col_a:
                fig_before = px.line(match['before'], x=date_col, y=target_col, 
                                    title="The Similar Period")
                fig_before.update_layout(template='plotly_dark', height=250, showlegend=False)
                st.plotly_chart(fig_before, width='stretch')
            
            with col_b:
                fig_after = px.line(match['after'], x=date_col, y=target_col,
                                   title="What Happened After")
                fig_after.update_layout(template='plotly_dark', height=250, showlegend=False)
                st.plotly_chart(fig_after, width='stretch')
        
        avg_change = np.mean([m['change_pct'] for m in matches])
        st.success(f"""
        **Historical Pattern:** When conditions looked like this in the past, 
        {target_col} changed by an average of **{avg_change:+.1f}%** in the following month.
        """)
    else:
        st.info("Not enough historical data to find similar periods.")

# TAB 3: Risk Scenarios
with tab3:
    st.markdown("### ⚠️ \"What would have to change for this forecast to be wrong?\"")
    
    scenarios = generate_risk_scenarios(future_preds['mean'].mean(), 
                                       {d: recent[d].mean() for d in driver_cols},
                                       driver_cols)
    
    # Calculate impact for each scenario
    scenario_results = []
    for scenario in scenarios:
        # Create scenario data
        scenario_df = future_df.copy()
        
        if scenario['driver'] != 'all':
            base = recent[scenario['driver']].mean()
            scenario_df[scenario['driver']] = base * (1 + scenario['change']/100)
        else:
            for d in driver_cols:
                base = recent[d].mean()
                scenario_df[d] = base * (1 + scenario['change']/100)
        
        # Recompute features
        for driver in driver_cols:
            if scenario['driver'] == driver or scenario['driver'] == 'all':
                base_val = recent[driver].mean()
                scenario_df[driver] = base_val * (1 + scenario['change']/100)
        
        X_scenario = scenario_df[feature_cols]
        scenario_pred = ensemble.predict_with_confidence(X_scenario)
        
        impact = ((scenario_pred['mean'].mean() - future_preds['mean'].mean()) 
                 / future_preds['mean'].mean() * 100)
        
        scenario_results.append({
            'name': scenario['name'],
            'type': scenario['type'],
            'likelihood': scenario['likelihood'],
            'impact': impact,
            'forecast': scenario_pred['mean'].mean()
        })
    
    # Display as cards
    for result in sorted(scenario_results, key=lambda x: abs(x['impact']), reverse=True):
        if 'downside' in result['type']:
            st.markdown(f"""
            <div class="risk-alert">
                <h4>🔻 {result['name']}</h4>
                <p><strong>Likelihood:</strong> {result['likelihood']}</p>
                <p><strong>Impact on Forecast:</strong> {result['impact']:.1f}%</p>
                <p><strong>New Forecast:</strong> {result['forecast']:.0f} (vs {future_preds['mean'].mean():.0f} baseline)</p>
            </div>
            """, unsafe_allow_html=True)
        elif result['impact'] > 5:
            st.markdown(f"""
            <div class="confidence-high">
                <h4>🔺 {result['name']}</h4>
                <p><strong>Likelihood:</strong> {result['likelihood']}</p>
                <p><strong>Impact on Forecast:</strong> +{result['impact']:.1f}%</p>
                <p><strong>New Forecast:</strong> {result['forecast']:.0f} (vs {future_preds['mean'].mean():.0f} baseline)</p>
            </div>
            """, unsafe_allow_html=True)

# TAB 4: Forecast Drivers
with tab4:
    st.markdown("### 🎯 Forecast Drivers")
    
    importance = ensemble.get_feature_importance()
    
    # Translate technical features
    def translate_feature(feat):
        if feat in driver_cols:
            return feat
        elif 'Lag_1' in feat:
            return 'Yesterday'
        elif 'Lag_7' in feat:
            return 'Last Week'
        elif 'Lag_28' in feat:
            return 'Last Month'
        elif 'RollingMean' in feat:
            return 'Recent Trend'
        elif 'RollingStd' in feat:
            return 'Volatility'
        elif 'Momentum' in feat:
            return 'Acceleration'
        elif 'DayOfWeek' in feat:
            return 'Day Pattern'
        elif 'Month' in feat:
            return 'Seasonal Pattern'
        else:
            return feat
    
    importance['Business_Name'] = importance['Feature'].apply(translate_feature)
    
    # Group and aggregate
    importance_grouped = importance.groupby('Business_Name')['Importance'].sum().reset_index()
    importance_grouped = importance_grouped.sort_values('Importance', ascending=False).head(8)
    
    fig = px.bar(importance_grouped, x='Importance', y='Business_Name', 
                 orientation='h', color='Importance',
                 color_continuous_scale='Teal')
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False,
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig, width='stretch')
    
    top_driver = importance_grouped.iloc[0]['Business_Name']
    st.success(f"""
    **Bottom Line:** The model pays most attention to **{top_driver}**. 
    If you want to influence the outcome, start there.
    """)
    
    # Model agreement visualization
    st.markdown("### How Much Do the Models Agree?")
    
    agreement_pct = 100 - (future_preds['model_std'].mean() / future_preds['mean'].mean() * 100)
    
    fig_agree = go.Figure()
    
    for i, (name, preds) in enumerate(zip(['GradientBoost', 'RandomForest', 'Ridge'], 
                                          future_preds['model_predictions'])):
        fig_agree.add_trace(go.Scatter(
            x=future_dates, y=preds,
            name=name, mode='lines',
            line=dict(width=2),
            hovertemplate=f'{name}: %{{y:.0f}}<extra></extra>'
        ))
    
    fig_agree.update_layout(
        template='plotly_dark',
        height=350,
        title=f"Model Agreement: {agreement_pct:.0f}%"
    )
    st.plotly_chart(fig_agree, width='stretch')
    
    if agreement_pct > 90:
        st.success("✅ **High Agreement**: All models see the same pattern. This forecast is reliable.")
    elif agreement_pct > 75:
        st.warning("⚠️ **Moderate Agreement**: Some divergence between models. Check risk scenarios.")
    else:
        st.error("🔴 **Low Agreement**: Models disagree significantly. High uncertainty - plan for multiple outcomes.")
