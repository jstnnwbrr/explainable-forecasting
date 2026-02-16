import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, PassiveAggressiveRegressor, TheilSenRegressor
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
import sys
import subprocess

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(page_title="Explainable Forecasting", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA; }
    .confidence-high {
        background: linear-gradient(135deg, #1e3a20, #2d5a2f);
        padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #3a3520, #5a4f2f);
        padding: 20px; border-radius: 10px; border-left: 5px solid #FFA726;
    }
    .confidence-low {
        background: linear-gradient(135deg, #3a2020, #5a2f2f);
        padding: 20px; border-radius: 10px; border-left: 5px solid #EF5350;
    }
    .precedent-card {
        background-color: #1a1f2e; padding: 15px; border-radius: 8px;
        border: 2px solid #3b82f6; margin: 10px 0;
    }
    .risk-alert {
        background-color: #2d1b1b; padding: 15px; border-radius: 8px;
        border-left: 4px solid #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(file):
    """Load CSV or Excel"""
    if file is None:
        return None
    
    try:
        name = file.name.lower()
        df = pd.read_excel(file) if name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
        
        # Auto-convert dates
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
    except Exception as e:
        st.error(f"Error loading: {e}")
        return None

# =============================================================================
# INTELLIGENT SEASONALITY DETECTION
# =============================================================================

def detect_seasonality(series, date_col_freq='D'):
    """
    Detect dominant seasonality using spectral analysis.
    Returns best period for decomposition.
    """
    try:
        # Remove trend via differencing
        detrended = np.diff(series.dropna())
        
        # FFT
        n = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(n, 1)
        
        # Power spectrum
        power = np.abs(yf[:n//2])**2
        freqs = xf[:n//2]
        
        # Find peaks
        peaks, _ = signal.find_peaks(power, height=np.mean(power))
        
        if len(peaks) == 0:
            # Default to weekly for daily data
            return 7 if date_col_freq == 'D' else 12
        
        # Get dominant frequency
        dominant_idx = peaks[np.argmax(power[peaks])]
        dominant_freq = freqs[dominant_idx]
        
        if dominant_freq == 0:
            return 7 if date_col_freq == 'D' else 12
        
        # Convert frequency to period
        period = int(1 / dominant_freq)
        
        # Reasonable bounds
        if date_col_freq == 'D':
            period = np.clip(period, 7, 365)  # Weekly to yearly
        else:
            period = np.clip(period, 4, 52)  # Quarterly to yearly
        
        return period
    
    except:
        # Fallback
        return 7 if date_col_freq == 'D' else 12

def decompose_series(df, date_col, target_col):
    """Decompose with intelligent seasonality detection"""
    try:
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        series = df_sorted[target_col].values
        
        # Detect seasonality
        period = detect_seasonality(pd.Series(series))
        
        st.info(f"🔍 Detected seasonality period: {period} (likely {'weekly' if period == 7 else 'monthly' if period == 30 else 'quarterly' if period == 90 else 'yearly' if period == 365 else 'custom'})")
        
        # STL decomposition
        stl = STL(series, seasonal=period, robust=True)
        result = stl.fit()
        
        return pd.DataFrame({
            date_col: df_sorted[date_col],
            'Observed': series,
            'Trend': result.trend,
            'Seasonal': result.seasonal,
            'Residual': result.resid
        }), period
    
    except Exception as e:
        st.warning(f"Decomposition failed: {e}")
        return None, None

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, date_col, target_col, driver_cols):
    """Create comprehensive features"""
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    
    # Calendar
    df['DayOfWeek'] = df[date_col].dt.dayofweek
    df['Month'] = df[date_col].dt.month
    df['Quarter'] = df[date_col].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['DayOfMonth'] = df[date_col].dt.day
    df['DayOfYear'] = df[date_col].dt.dayofyear
    
    # Cyclical
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Lags
    for lag in [1, 7, 14, 28]:
        df[f'Lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling
    for window in [7, 14, 28]:
        df[f'RollingMean_{window}'] = df[target_col].rolling(window).mean().shift(1)
        df[f'RollingStd_{window}'] = df[target_col].rolling(window).std().shift(1)
    
    # Momentum
    df['Momentum_7'] = df[target_col].diff(7)
    df['Acceleration_7'] = df['Momentum_7'].diff(7)
    
    # EMA
    df['EMA_7'] = df[target_col].ewm(span=7).mean().shift(1)
    df['EMA_28'] = df[target_col].ewm(span=28).mean().shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    engineered = [
        'DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'DayOfMonth', 'DayOfYear',
        'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos',
        'Lag_1', 'Lag_7', 'Lag_14', 'Lag_28',
        'RollingMean_7', 'RollingMean_14', 'RollingMean_28',
        'RollingStd_7', 'RollingStd_14', 'RollingStd_28',
        'Momentum_7', 'Acceleration_7',
        'EMA_7', 'EMA_28'
    ]
    
    available_drivers = [d for d in driver_cols if d in df.columns]
    
    return df, available_drivers + engineered

def create_future_features(future_df, recent_history, date_col, target_col, driver_cols, feature_cols):
    """
    CRITICAL: Properly reconstruct ALL features for future dates.
    This fixes the KeyError bug.
    """
    # Calendar features (can be calculated from dates)
    future_df['DayOfWeek'] = future_df[date_col].dt.dayofweek
    future_df['Month'] = future_df[date_col].dt.month
    future_df['Quarter'] = future_df[date_col].dt.quarter
    future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
    future_df['DayOfMonth'] = future_df[date_col].dt.day
    future_df['DayOfYear'] = future_df[date_col].dt.dayofyear
    
    # Cyclical encoding
    future_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
    future_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
    future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
    future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
    
    # For lag features, use recent history
    # This is an approximation for forecasting
    for lag in [1, 7, 14, 28]:
        # Use rolling average from recent history as proxy
        if len(recent_history) >= lag:
            future_df[f'Lag_{lag}'] = recent_history[target_col].iloc[-lag:].mean()
        else:
            future_df[f'Lag_{lag}'] = recent_history[target_col].mean()
    
    # Rolling statistics - use recent values
    for window in [7, 14, 28]:
        if len(recent_history) >= window:
            future_df[f'RollingMean_{window}'] = recent_history[target_col].tail(window).mean()
            future_df[f'RollingStd_{window}'] = recent_history[target_col].tail(window).std()
        else:
            future_df[f'RollingMean_{window}'] = recent_history[target_col].mean()
            future_df[f'RollingStd_{window}'] = recent_history[target_col].std()
    
    # Momentum and acceleration - use recent trends
    if 'Momentum_7' in recent_history.columns:
        future_df['Momentum_7'] = recent_history['Momentum_7'].tail(7).mean()
        future_df['Acceleration_7'] = recent_history['Acceleration_7'].tail(7).mean()
    else:
        # Fallback
        future_df['Momentum_7'] = 0
        future_df['Acceleration_7'] = 0
    
    # EMA - use recent values
    if 'EMA_7' in recent_history.columns:
        future_df['EMA_7'] = recent_history['EMA_7'].iloc[-1]
        future_df['EMA_28'] = recent_history['EMA_28'].iloc[-1]
    else:
        future_df['EMA_7'] = recent_history[target_col].tail(7).mean()
        future_df['EMA_28'] = recent_history[target_col].tail(28).mean()
    
    # Ensure all feature_cols exist
    for col in feature_cols:
        if col not in future_df.columns:
            # If still missing, use mean from recent history
            if col in recent_history.columns:
                future_df[col] = recent_history[col].mean()
            else:
                future_df[col] = 0
    
    return future_df

# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class ProductionEnsemble:
    """10-model ensemble with confidence intervals"""
    
    def __init__(self):
        self.models = {}
        self.model_predictions = {}
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_names = None
        self.historical_errors = []
        self.target_col = None
        
    def fit(self, df, date_col, target_col, feature_cols):
        """Train all models"""
        self.target_col = target_col
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        dates = df[date_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        all_preds = []
        
        # 6. ElasticNet
        try:
            with st.spinner("Training ElasticNet..."):
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
                model.fit(X_scaled, y)
                pred = model.predict(X_scaled)
                self.models['ElasticNet'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"ElasticNet failed: {e}")
        
        # 7. PassiveAggressive
        try:
            with st.spinner("Training PassiveAggressive..."):
                model = PassiveAggressiveRegressor(max_iter=1000, random_state=42)
                model.fit(X_scaled, y)
                pred = model.predict(X_scaled)
                self.models['PassiveAggressive'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"PassiveAggressive failed: {e}")
        
        # 8. Polynomial Regression
        try:
            with st.spinner("Training Polynomial..."):
                X_subset = X[:, :min(5, X.shape[1])]
                X_poly = self.poly_features.fit_transform(X_subset)
                
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_poly, y)
                pred = model.predict(X_poly)
                self.models['Polynomial'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"Polynomial failed: {e}")
        
        # 9. TheilSen
        try:
            with st.spinner("Training TheilSen..."):
                sample_size = min(500, len(X))
                idx = np.random.choice(len(X), sample_size, replace=False)
                
                model = TheilSenRegressor(random_state=42, max_iter=300, n_jobs=-1)
                model.fit(X_scaled[idx], y[idx])
                pred = model.predict(X_scaled)
                self.models['TheilSen'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"TheilSen failed: {e}")
        
        # 10. MLP
        try:
            with st.spinner("Training MLP..."):
                model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    early_stopping=True,
                    random_state=42
                )
                model.fit(X_scaled, y)
                pred = model.predict(X_scaled)
                self.models['MLP'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"MLP failed: {e}")
        
        # 4. Auto-ARIMA
        try:
            with st.spinner("Training Auto-ARIMA..."):
                model = auto_arima(
                    y,
                    seasonal=True, m=7,
                    max_p=3, max_q=3,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                pred = model.predict_in_sample()
                self.models['AutoARIMA'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"Auto-ARIMA failed: {e}")
        
        # 5. Exponential Smoothing
        try:
            with st.spinner("Training ExpSmoothing..."):
                y_pos = y - y.min() + 1 if y.min() <= 0 else y
                
                model = ExponentialSmoothing(
                    y_pos,
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add'
                )
                fit = model.fit()
                pred = fit.fittedvalues
                
                if y.min() <= 0:
                    pred = pred + y.min() - 1
                
                self.models['ExpSmoothing'] = fit
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"ExpSmoothing failed: {e}")
        
        # 3. Prophet
        try:
            with st.spinner("Training Prophet..."):
                prophet_df = pd.DataFrame({'ds': dates, 'y': y})
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                model.fit(prophet_df)
                forecast = model.predict(prophet_df)
                pred = forecast['yhat'].values
                
                self.models['Prophet'] = model
                all_preds.append(pred)
        except Exception as e:
            st.warning(f"Prophet failed: {e}")
        
        # Calculate ensemble
        if all_preds:
            all_preds = np.array(all_preds)
            ensemble_pred = np.mean(all_preds, axis=0)
            self.historical_errors = y - ensemble_pred
            
            for i, name in enumerate(list(self.models.keys())[:len(all_preds)]):
                self.model_predictions[name] = all_preds[i]
        
        st.success(f"✓ {len(self.models)} models trained successfully")
        
        return self
    
    def predict_with_confidence(self, future_df, feature_cols):
        """Predict with confidence intervals"""
        X_future = future_df[feature_cols].values
        X_future_scaled = self.scaler.transform(X_future)
        
        predictions = []
        
        # Regression models
        for name in ['ElasticNet', 'PassiveAggressive', 'TheilSen', 'MLP']:
            if name in self.models:
                pred = self.models[name].predict(X_future_scaled)
                predictions.append(pred)
        
        # Polynomial
        if 'Polynomial' in self.models:
            X_subset = X_future[:, :min(5, X_future.shape[1])]
            X_poly = self.poly_features.transform(X_subset)
            pred = self.models['Polynomial'].predict(X_poly)
            predictions.append(pred)
        
        # Time series
        if 'AutoARIMA' in self.models:
            pred = self.models['AutoARIMA'].predict(n_periods=len(future_df))
            predictions.append(pred)
        
        if 'ExpSmoothing' in self.models:
            pred = self.models['ExpSmoothing'].forecast(steps=len(future_df))
            predictions.append(pred)
        
        if 'Prophet' in self.models:
            prophet_future = pd.DataFrame({'ds': future_df[future_df.columns[0]].values})
            forecast = self.models['Prophet'].predict(prophet_future)
            pred = forecast['yhat'].values
            predictions.append(pred)
        
        if not predictions:
            st.error("No models succeeded!")
            return None
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        model_std = np.std(predictions, axis=0)
        error_std = np.std(self.historical_errors)
        
        total_std = np.sqrt(model_std**2 + error_std**2)
        
        return {
            'mean': mean_pred,
            'model_std': model_std,
            'error_std': error_std,
            'ci_68': (mean_pred - total_std, mean_pred + total_std),
            'ci_95': (mean_pred - 2*total_std, mean_pred + 2*total_std),
            'model_predictions': predictions
        }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_similar_periods(current, historical, date_col, target_col, n=3):
    """Find similar historical periods"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    window = len(current)
    if len(historical) < window * 2:
        return []
    
    windows, indices = [], []
    
    for i in range(len(historical) - window - 30):
        windows.append(historical.iloc[i:i+window][target_col].values)
        indices.append(i)
    
    if not windows:
        return []
    
    windows = np.array(windows)
    current_pattern = current[target_col].values.reshape(1, -1)
    
    similarities = cosine_similarity(current_pattern, windows)[0]
    top_idx = similarities.argsort()[-n:][::-1]
    
    matches = []
    for idx in top_idx:
        start = indices[idx]
        match_date = historical.iloc[start][date_col]
        before = historical.iloc[start:start+window]
        after = historical.iloc[start+window:start+window+30]
        
        if len(after) > 0:
            matches.append({
                'start_date': match_date,
                'similarity': similarities[idx],
                'before': before,
                'after': after,
                'avg_before': before[target_col].mean(),
                'avg_after': after[target_col].mean(),
                'change_pct': ((after[target_col].mean() - before[target_col].mean()) 
                              / before[target_col].mean() * 100)
            })
    
    return matches

def generate_risk_scenarios(driver_cols):
    """Generate risk scenarios"""
    scenarios = []
    
    for driver in driver_cols:
        scenarios.extend([
            {'name': f'{driver} ↓20%', 'driver': driver, 'change': -20, 
             'type': 'downside', 'likelihood': 'Medium'},
            {'name': f'{driver} ↑20%', 'driver': driver, 'change': 20,
             'type': 'upside', 'likelihood': 'Medium'}
        ])
    
    scenarios.append({
        'name': 'Market Shock', 'driver': 'all', 'change': -30,
        'type': 'extreme_downside', 'likelihood': 'Low'
    })
    
    return scenarios

# =============================================================================
# MAIN APP
# =============================================================================

st.title("Explainable Forecasting")

# Sidebar
with st.sidebar:
    st.title("📂 Upload")
    
    uploaded = st.file_uploader("CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded is None:
        st.error("⚠️ Upload required")
        st.info("Need: date column + numeric target + optional drivers")
        st.stop()
    
    raw_df = load_data(uploaded)
    
    if raw_df is None:
        st.stop()
    
    st.success(f"✓ {len(raw_df)} rows")
    
    # Columns
    all_cols = list(raw_df.columns)
    date_cols = [c for c in all_cols if raw_df[c].dtype == 'datetime64[ns]']
    
    if not date_cols:
        date_cols = [c for c in all_cols if 'date' in c.lower()]
    
    if not date_cols:
        st.error("No date column!")
        st.stop()
    
    date_col = st.selectbox("Date", date_cols)
    
    try:
        raw_df[date_col] = pd.to_datetime(raw_df[date_col])
    except:
        st.error("Can't parse dates")
        st.stop()
    
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns!")
        st.stop()
    
    target_col = st.selectbox("Target", numeric_cols)
    
    driver_cols = st.multiselect(
        "Drivers (optional)",
        [c for c in numeric_cols if c != target_col]
    )
    
    st.markdown("---")
    st.markdown("### 🎲 Scenario")
    
    scenario_changes = {}
    if driver_cols:
        for driver in driver_cols:
            scenario_changes[driver] = st.slider(
                driver, -50, 50, 0, key=f"scn_{driver}"
            )

# Feature engineering
with st.spinner("Engineering features..."):
    processed_df, feature_cols = engineer_features(raw_df, date_col, target_col, driver_cols)

st.info(f"📊 {len(processed_df)} samples ready")

# Split
train_size = len(processed_df) - 60
train_df = processed_df.iloc[:train_size]
test_df = processed_df.iloc[train_size:]

# Train
ensemble = ProductionEnsemble()

with st.spinner("Training models..."):
    ensemble.fit(train_df, date_col, target_col, feature_cols)

# Evaluate
test_preds = ensemble.predict_with_confidence(test_df, feature_cols)

if test_preds is None:
    st.stop()

mape = mean_absolute_percentage_error(test_df[target_col], test_preds['mean'])
mae = mean_absolute_error(test_df[target_col], test_preds['mean'])

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{(1-mape)*100:.1f}%")
with col2:
    conf = 100 - (test_preds['model_std'].mean() / test_preds['mean'].mean() * 100)
    st.metric("Confidence", f"{conf:.0f}/100")
with col3:
    st.metric("Avg Error", f"±{mae:.0f}")
with col4:
    st.metric("Models", len(ensemble.models))

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast",
    "🔬 Trend & Seasonality",
    "🔍 Historical Precedent",
    "⚠️ Risk"
])

# Generate forecast
last_date = processed_df[date_col].max()
future_dates = pd.date_range(last_date + timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame({date_col: future_dates})

# Apply scenarios to drivers
recent = processed_df.tail(60)
for driver in driver_cols:
    base = recent[driver].mean()
    pct = scenario_changes.get(driver, 0)
    future_df[driver] = base * (1 + pct/100)

# CRITICAL: Properly create all features
future_df = create_future_features(future_df, recent, date_col, target_col, driver_cols, feature_cols)

# Predict
future_preds = ensemble.predict_with_confidence(future_df, feature_cols)

# TAB 1
with tab1:
    st.markdown("### Forecast")
    
    fig = go.Figure()
    
    hist = processed_df.tail(90)
    fig.add_trace(go.Scatter(
        x=hist[date_col], y=hist[target_col],
        name='History', line=dict(color='gray', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['mean'],
        name='Forecast', line=dict(color='#00CC96', width=4)
    ))
    
    # CI bands
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_68'][1],
        fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_68'][0],
        fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
        name='68% Confidence', fillcolor='rgba(0,204,150,0.3)'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_95'][1],
        fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds['ci_95'][0],
        fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
        name='95% Confidence', fillcolor='rgba(0,204,150,0.15)'
    ))
    
    fig.update_layout(template='plotly_dark', height=500)
    st.plotly_chart(fig, width='stretch')
    
    avg_forecast = future_preds['mean'].mean()
    avg_hist = recent[target_col].tail(30).mean()
    pct_change = (avg_forecast - avg_hist) / avg_hist * 100
    
    ci_width = (future_preds['ci_68'][1].mean() - future_preds['ci_68'][0].mean())
    conf_level = "HIGH" if ci_width < avg_forecast * 0.2 else "MEDIUM" if ci_width < avg_forecast * 0.4 else "LOW"
    
    css = "confidence-high" if conf_level == "HIGH" else "confidence-medium" if conf_level == "MEDIUM" else "confidence-low"
    emoji = "✅" if conf_level == "HIGH" else "⚠️" if conf_level == "MEDIUM" else "🔴"
    
    st.markdown(f"""
    <div class="{css}">
        <h3>{emoji} Confidence: {conf_level}</h3>
        <p>Forecast: {avg_forecast:.0f} ({pct_change:+.1f}% change)</p>
        <ul>
            <li>68% chance within ±{ci_width/2:.0f}</li>
            <li>95% chance within ±{(future_preds['ci_95'][1].mean() - future_preds['ci_95'][0].mean())/2:.0f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 2
with tab2:
    st.markdown("### Decomposition")
    
    decomp, period = decompose_series(processed_df, date_col, target_col)
    
    if decomp is not None:
        fig = go.Figure()
        
        for comp, color in zip(['Observed', 'Trend', 'Seasonal', 'Residual'],
                               ['white', '#00CC96', '#FFA726', '#EF553B']):
            fig.add_trace(go.Scatter(
                x=decomp[date_col], y=decomp[comp],
                name=comp, line=dict(color=color)
            ))
        
        fig.update_layout(template='plotly_dark', height=600)
        st.plotly_chart(fig, width='stretch')

# TAB 3
with tab3:
    st.markdown("### Historical Precedent")
    
    current = processed_df.tail(30)
    matches = find_similar_periods(current, processed_df.head(train_size), date_col, target_col)
    
    if matches:
        for i, m in enumerate(matches):
            st.markdown(f"""
            <div class="precedent-card">
                <h4>Match #{i+1}: {m['start_date'].strftime('%B %Y')} ({m['similarity']*100:.0f}% similar)</h4>
                <p>{m['avg_before']:.0f} → {m['avg_after']:.0f} ({m['change_pct']:+.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.line(m['before'], x=date_col, y=target_col)
                fig.update_layout(template='plotly_dark', height=250)
                st.plotly_chart(fig, width='stretch')
            with col_b:
                fig = px.line(m['after'], x=date_col, y=target_col)
                fig.update_layout(template='plotly_dark', height=250)
                st.plotly_chart(fig, width='stretch')
        
        avg = np.mean([m['change_pct'] for m in matches])
        st.success(f"Historical avg: **{avg:+.1f}%**")

# TAB 4
with tab4:
    st.markdown("### Risk Scenarios")
    
    if not driver_cols:
        st.warning("No drivers selected")
    else:
        scenarios = generate_risk_scenarios(driver_cols)
        
        for scenario in scenarios[:5]:
            scenario_df = future_df.copy()
            
            if scenario['driver'] != 'all':
                base = recent[scenario['driver']].mean()
                scenario_df[scenario['driver']] = base * (1 + scenario['change']/100)
            else:
                for d in driver_cols:
                    base = recent[d].mean()
                    scenario_df[d] = base * (1 + scenario['change']/100)
            
            # Recreate features for scenario
            scenario_df = create_future_features(scenario_df, recent, date_col, target_col, driver_cols, feature_cols)
            
            scenario_pred = ensemble.predict_with_confidence(scenario_df, feature_cols)
            impact = ((scenario_pred['mean'].mean() - future_preds['mean'].mean()) 
                     / future_preds['mean'].mean() * 100)
            
            if 'downside' in scenario['type']:
                st.markdown(f"""
                <div class="risk-alert">
                    <h4>{scenario['name']}</h4>
                    <p>Impact: {impact:.1f}% | New: {scenario_pred['mean'].mean():.0f}</p>
                </div>
                """, unsafe_allow_html=True)
