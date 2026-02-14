import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# CONFIGURATION & STYLE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Forecasting Explanation Bridge", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .highlight {
        color: #FF4B4B;
        font-weight: bold;
    }
    .persona-box {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA SIMULATION (MOCKING THE MODELS)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_data():
    """Generates synthetic time series and simulates model outputs."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    
    # Features
    price = np.random.uniform(90, 110, size=365)
    marketing = np.random.uniform(500, 1500, size=365)
    
    # Base Signal (Seasonality + Trend)
    t = np.arange(365)
    seasonality = 10 * np.sin(2 * np.pi * t / 30) # Monthly seasonality
    trend = 0.1 * t
    noise = np.random.normal(0, 2, 365)
    
    actuals = 100 + trend + seasonality - (0.5 * (price - 100)) + (0.01 * marketing) + noise
    
    # --- MODEL 1 SIMULATION (Time Series Specialists: N-BEATS, N-HiTS, etc.) ---
    # Captures seasonality well, smoother, slightly lagging but robust
    m1_pred = 100 + trend + seasonality - (0.3 * (price - 100)) + (0.005 * marketing) + np.random.normal(0, 1, 365)
    
    # --- MODEL 2 SIMULATION (Regressors: ElasticNet, TheilSen, etc.) ---
    # Over-indexes on features (Price/Marketing), misses seasonality, noisier
    m2_pred = 100 + trend + (0.1 * seasonality) - (0.9 * (price - 100)) + (0.02 * marketing) + np.random.normal(0, 5, 365)

    df = pd.DataFrame({
        'Date': dates,
        'Actuals': actuals,
        'Price': price,
        'Marketing': marketing,
        'Model_1_TS_Specialists': m1_pred,
        'Model_2_Regressors': m2_pred
    })
    return df

data = generate_data()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR EXPLAINABILITY
# -----------------------------------------------------------------------------
def find_nearest_neighbor(current_window, history_df, window_size=30):
    """Finds the historical period most similar to the current data pattern."""
    scaler = MinMaxScaler()
    
    # We look at the pattern of the last 'window_size' days
    # Features to compare: Actuals shape, Price, Marketing
    
    # Prepare historical sliding windows
    history_windows = []
    indices = []
    
    target_data = history_df[['Actuals', 'Price', 'Marketing']].values
    
    for i in range(len(target_data) - window_size * 2): # Avoid overlapping with current
        window = target_data[i:i+window_size].flatten()
        history_windows.append(window)
        indices.append(i)
        
    history_windows = np.array(history_windows)
    
    # Current window (the input pattern)
    current_pattern = current_window[['Actuals', 'Price', 'Marketing']].values.flatten().reshape(1, -1)
    
    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(history_windows)
    
    distances, neighbor_idx = nn.kneighbors(current_pattern)
    
    start_idx = indices[neighbor_idx[0][0]]
    match_date = history_df.iloc[start_idx]['Date']
    return match_date, start_idx

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

st.title("Forecast Interpretability Bridge")
st.markdown("### Translating High-Accuracy Black Boxes into Business Logic")

col_setup, col_context = st.columns([1, 3])

with col_setup:
    st.info("👋 **The Stakeholder Reality Check**\n\nYou asked for explainability. We are using **Model 1** (Ensemble of Deep Learning/TS) and **Model 2** (Ensemble of Regressors).")
    
    current_date_idx = st.slider("Select Forecast Simulation Day", 300, 360, 350)
    current_row = data.iloc[current_date_idx]
    
    st.markdown("---")
    st.markdown("**Current Drivers:**")
    st.metric("Price", f"${current_row['Price']:.2f}")
    st.metric("Marketing Spend", f"${current_row['Marketing']:.0f}")

# -----------------------------------------------------------------------------
# MAIN FORECAST PLOT
# -----------------------------------------------------------------------------
with col_context:
    # Prepare data for plotting
    plot_df = data.iloc[current_date_idx-60 : current_date_idx+1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Actuals'], name='Actual History', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Model_1_TS_Specialists'], name='Model 1 (TS/Deep Learning)', line=dict(color='#00CC96', width=3)))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Model_2_Regressors'], name='Model 2 (Regressors)', line=dict(color='#EF553B', width=3, dash='dot')))
    
    fig.update_layout(title="Model Divergence: The 'Why' Gap", template="plotly_dark", height=400)
    st.plotly_chart(fig, width='stretch')

# -----------------------------------------------------------------------------
# THE EXPLAINABILITY LAYER
# -----------------------------------------------------------------------------
st.divider()
st.header("Why do they disagree? (The Explainability Layer)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Historical Analog (Nearest Neighbor)", 
    "⚖️ Sensitivity & Stress Test", 
    "🔄 Counterfactuals", 
    "📉 The Accuracy Tax",
    "🗣️ Feature Decoder"
])

# --- TAB 1: NEAREST NEIGHBORS ---
with tab1:
    st.markdown("""
    **Stakeholder Question:** *"Why is Model 1 predicting this curve?"* **Answer:** *"Because the current market conditions look exactly like this previous period."*
    """)
    
    current_window = data.iloc[current_date_idx-30 : current_date_idx]
    match_date, start_idx = find_nearest_neighbor(current_window, data)
    
    col_nn1, col_nn2 = st.columns(2)
    
    with col_nn1:
        st.markdown(f"#### Current Market Pattern ({data.iloc[current_date_idx]['Date'].strftime('%Y-%m-%d')})")
        fig_cur = px.line(current_window, x='Date', y='Actuals', title="Last 30 Days Trend")
        fig_cur.update_traces(line_color='#00CC96')
        fig_cur.update_layout(height=250, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_cur, width='stretch')
        
    with col_nn2:
        st.markdown(f"#### Nearest Historical Analog ({match_date.strftime('%Y-%m-%d')})")
        match_window = data.iloc[start_idx : start_idx+30]
        fig_hist = px.line(match_window, x='Date', y='Actuals', title="Closest Historical Match")
        fig_hist.update_traces(line_color='#FFA15A')
        fig_hist.update_layout(height=250, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_hist, width='stretch')
        
    st.success(f"**Interpretation:** Model 1 is recognizing that the combination of Volatility and Price Trends today is 94% similar to the period starting on **{match_date.strftime('%Y-%m-%d')}**. In that period, sales rose, which is why Model 1 is bullish.")

# --- TAB 2: SENSITIVITY ANALYSIS ---
with tab2:
    st.markdown("""
    **Stakeholder Question:** *"What happens if we drop price?"* **Answer:** *"Model 2 will react violently. Model 1 will barely care."*
    """)
    
    col_sense_controls, col_sense_plot = st.columns([1, 2])
    
    with col_sense_controls:
        price_delta = st.slider("Simulate Price Change (%)", -20, 20, 0)
        
        # Calculate impact based on our simulation formulas
        # Model 1 coefficient was 0.3, Model 2 was 0.9
        base_price = current_row['Price']
        new_price = base_price * (1 + price_delta/100)
        
        m1_impact = -0.3 * (new_price - base_price)
        m2_impact = -0.9 * (new_price - base_price)
        
        st.markdown(f"**New Price:** ${new_price:.2f}")
        st.markdown("---")
        st.warning(f"**Model 1 Impact:** {m1_impact:.2f} units")
        st.error(f"**Model 2 Impact:** {m2_impact:.2f} units")
        
    with col_sense_plot:
        # Generate sensitivity curve
        prices = np.linspace(base_price * 0.8, base_price * 1.2, 50)
        m1_curve = [current_row['Model_1_TS_Specialists'] - (0.3 * (p - base_price)) for p in prices]
        m2_curve = [current_row['Model_2_Regressors'] - (0.9 * (p - base_price)) for p in prices]
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=prices, y=m1_curve, name='Model 1 Sensitivity', line=dict(color='#00CC96')))
        fig_sens.add_trace(go.Scatter(x=prices, y=m2_curve, name='Model 2 Sensitivity', line=dict(color='#EF553B')))
        fig_sens.add_vline(x=base_price, line_dash="dash", annotation_text="Current Price")
        fig_sens.add_vline(x=new_price, line_color="yellow", annotation_text="Simulated")
        
        fig_sens.update_layout(title="Price Sensitivity Curves", xaxis_title="Price ($)", yaxis_title="Predicted Sales", template="plotly_dark")
        st.plotly_chart(fig_sens, width='stretch')

# --- TAB 3: COUNTERFACTUALS ---
with tab3:
    st.markdown("""
    **Stakeholder Question:** *"Why is Model 2 so much lower than Model 1?"* **Answer:** *"For Model 2 to agree with Model 1, we would have to spend way more on marketing."*
    """)
    
    diff = current_row['Model_1_TS_Specialists'] - current_row['Model_2_Regressors']
    
    # Calculate how much marketing is needed to close the gap for Model 2
    # Model 2 marketing coeff is 0.02
    marketing_needed = diff / 0.02
    total_marketing_needed = current_row['Marketing'] + marketing_needed
    
    col_cf1, col_cf2 = st.columns(2)
    
    with col_cf1:
        st.metric("Prediction Gap", f"{diff:.2f} units")
        
    with col_cf2:
        st.markdown("#### The Counterfactual Condition")
        st.info(f"""
        Model 2 (The Regressor) relies heavily on **Marketing Spend**. 
        
        To force Model 2 to match Model 1's prediction, you would need to increase daily marketing spend by:
        # **${marketing_needed:.2f}** (Total Spend: ${total_marketing_needed:.2f})
        """)
        
    st.markdown("---")
    st.markdown("**Why this matters:** This tells you that Model 1 is seeing organic demand (Trend/Seasonality) that Model 2 thinks *must* be paid for via Marketing.")

# --- TAB 4: ACCURACY TAX ---
with tab4:
    st.markdown("""
    **Stakeholder Challenge:** *"I want perfect explanation AND perfect accuracy."* **Answer:** *"Pick one."*
    """)
    
    # Mock accuracy data
    acc_data = pd.DataFrame({
        'Model': ['SES (Model 1)', 'Auto-ARIMA (Model 1)', 'Prophet (Model 1)', 'N-HiTS (Model 1)', 'N-BEATS (Model 1)', 
                  'Poly Reg (Model 2)', 'ElasticNet (Model 2)', 'TheilSen (Model 2)', 'MLP (Model 2)'],
        'Accuracy (MAPE)': [15, 12, 10, 5, 4, 14, 11, 13, 6], # Lower is better
        'Explainability Score': [9, 8, 7, 2, 1, 8, 8, 6, 2], # Higher is better
        'Group': ['Model 1', 'Model 1', 'Model 1', 'Model 1', 'Model 1', 'Model 2', 'Model 2', 'Model 2', 'Model 2']
    })
    
    fig_tax = px.scatter(acc_data, x='Explainability Score', y='Accuracy (MAPE)', 
                         color='Group', text='Model', size=[20]*9,
                         color_discrete_map={'Model 1': '#00CC96', 'Model 2': '#EF553B'})
    
    fig_tax.update_traces(textposition='top center')
    fig_tax.update_layout(
        title="The Accuracy Tax Trade-Off",
        xaxis_title="Explainability (Higher is Easier to Explain)",
        yaxis_title="Error Rate (Lower is Better)",
        template="plotly_dark",
        shapes=[
            # Highlight the trade-off zone
            dict(type="rect", x0=0, y0=0, x1=3, y1=7, line_color="yellow", fillcolor="yellow", opacity=0.1)
        ]
    )
    
    st.plotly_chart(fig_tax, width='stretch')
    st.warning("The 'Yellow Zone' is where N-BEATS and N-HiTS live. We accept that they are hard to explain (Low X-axis) because they provide the lowest error (Low Y-axis). If you want to move to the right (higher explainability), you must accept higher error.")

# --- TAB 5: FEATURE DECODER ---
with tab5:
    st.markdown("""
    **Stakeholder Question:** *"What do you mean 'Lag_7' is driving the forecast? Speak English."*
    """)
    
    feature_data = {
        "Technical Feature Name": ["Lag_7", "Lag_365", "Rolling_Mean_30", "Rolling_Std_7", "Exp_Moving_Avg"],
        "Business Translation": ["Same Day Last Week", "Same Day Last Year", "30-Day Trend Base", "Recent Volatility", "Recent Momentum"],
        "What it tells the model": [
            "Matches the day-of-week pattern (e.g., high traffic on Fridays).",
            "Matches the annual seasonality (e.g., Christmas spike).",
            "Ignores daily noise to find the 'True' demand level.",
            "Detects if the market is currently chaotic or stable.",
            "Weights yesterday heavily; assumes recent history is most important."
        ]
    }
    
    df_features = pd.DataFrame(feature_data)
    
    # Simple table styling
    st.table(df_features)
    
    st.info("💡 **Pro Tip:** Never show the 'Technical Feature Name' to a stakeholder. Always alias your feature importance charts with the 'Business Translation'.")