import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

# =============================================================================
# Exchange Setup & Data Fetching Functions
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_exchange():
    """
    Create and return a ccxt exchange instance. Here we use Binance.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    exchange.load_markets()
    return exchange

exchange = get_exchange()

@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, timeframe, since=None, limit=None):
    """
    Fetch OHLCV data using ccxt for a given symbol, timeframe, since and limit.
    Returns a pandas DataFrame.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# =============================================================================
# Dashboard Page (Modified)
# =============================================================================

def dashboard_page():
    st.title("Crypto Dashboard - Quant Analysis Over 30 Days")
    st.write("Select crypto pairs to view detailed quant analysis metrics.")
    
    # -------------------------------------------------------------------------
    # Pair Selection (Primary and Optional Secondary)
    # -------------------------------------------------------------------------
    available_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'GALA/USDT', 'UNI/USDT']
    primary_pair = st.selectbox("Select Primary Crypto Pair", available_pairs, index=0)
    secondary_options = ["None"] + available_pairs
    secondary_pair = st.selectbox("Select Secondary Crypto Pair (Optional)", secondary_options, index=0)
    
    # -------------------------------------------------------------------------
    # Fetch Data for Last 30 Days
    # -------------------------------------------------------------------------
    now = exchange.milliseconds()
    thirty_days_ago = now - 30 * 24 * 60 * 60 * 1000
    df_primary = fetch_ohlcv(primary_pair, '1d', since=thirty_days_ago)
    if df_primary is None or df_primary.empty:
        st.error(f"No data available for {primary_pair}")
        return
    
    if secondary_pair != "None":
        df_secondary = fetch_ohlcv(secondary_pair, '1d', since=thirty_days_ago)
        if df_secondary is None or df_secondary.empty:
            st.warning(f"No data available for {secondary_pair}")
            df_secondary = None
    else:
        df_secondary = None

    # =========================================================================
    # Graph 1: Smooth Price Line Graph with Markers
    # =========================================================================
    st.subheader("1. Price Trend")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_primary['datetime'],
        y=df_primary['close'],
        mode='lines+markers',
        line_shape='spline',
        name=primary_pair
    ))
    if df_secondary is not None:
        fig_price.add_trace(go.Scatter(
            x=df_secondary['datetime'],
            y=df_secondary['close'],
            mode='lines+markers',
            line_shape='spline',
            name=secondary_pair
        ))
    fig_price.update_layout(
        title="Price Trend Over 30 Days",
        xaxis_title="Date",
        yaxis_title="Closing Price"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # =========================================================================
    # Graph 2: Detailed Probability Distribution of Daily Returns
    # =========================================================================
    st.subheader("2. Probability Distribution of Daily Returns")
    # Calculate daily returns (%) for primary
    returns_primary = df_primary['close'].pct_change().dropna() * 100
    last_return_primary = df_primary['close'].pct_change().iloc[-1] * 100
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Histogram(
        x=returns_primary,
        nbinsx=30,
        name=primary_pair,
        opacity=0.75
    ))
    if df_secondary is not None:
        returns_secondary = df_secondary['close'].pct_change().dropna() * 100
        last_return_secondary = df_secondary['close'].pct_change().iloc[-1] * 100
        fig_prob.add_trace(go.Histogram(
            x=returns_secondary,
            nbinsx=30,
            name=secondary_pair,
            opacity=0.75
        ))
    # Add vertical lines for today's return(s)
    fig_prob.add_vline(
        x=last_return_primary,
        line=dict(color='blue', dash='dash'),
        annotation_text=f"Today's Return: {last_return_primary:.2f}%",
        annotation_position="top left"
    )
    if df_secondary is not None:
        fig_prob.add_vline(
            x=last_return_secondary,
            line=dict(color='orange', dash='dash'),
            annotation_text=f"Today's Return: {last_return_secondary:.2f}%",
            annotation_position="top right"
        )
    fig_prob.update_layout(
        title="Probability Distribution of Daily Returns (%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # =========================================================================
    # Graph 3: Mean Reversion Analysis (Z-score of Closing Price)
    # =========================================================================
    st.subheader("3. Mean Reversion Analysis (Closing Price Z-score)")
    z_primary = (df_primary['close'] - df_primary['close'].mean()) / df_primary['close'].std()
    fig_mean = go.Figure()
    fig_mean.add_trace(go.Scatter(
        x=df_primary['datetime'],
        y=z_primary,
        mode='lines+markers',
        name=primary_pair
    ))
    if df_secondary is not None:
        z_secondary = (df_secondary['close'] - df_secondary['close'].mean()) / df_secondary['close'].std()
        fig_mean.add_trace(go.Scatter(
            x=df_secondary['datetime'],
            y=z_secondary,
            mode='lines+markers',
            name=secondary_pair
        ))
    fig_mean.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_mean.update_layout(
        title="Mean Reversion Analysis (Z-score of Closing Price)",
        xaxis_title="Date",
        yaxis_title="Z-score"
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    # =========================================================================
    # Graph 4: Standardization of Daily Returns (Z-score)
    # =========================================================================
    st.subheader("4. Standardization of Daily Returns (Z-score)")
    returns_primary_clean = df_primary['close'].pct_change().dropna()
    z_returns_primary = (returns_primary_clean - returns_primary_clean.mean()) / returns_primary_clean.std()
    fig_std = go.Figure()
    fig_std.add_trace(go.Scatter(
        x=df_primary['datetime'].iloc[1:],
        y=z_returns_primary,
        mode='lines+markers',
        name=primary_pair
    ))
    if df_secondary is not None:
        returns_secondary_clean = df_secondary['close'].pct_change().dropna()
        z_returns_secondary = (returns_secondary_clean - returns_secondary_clean.mean()) / returns_secondary_clean.std()
        fig_std.add_trace(go.Scatter(
            x=df_secondary['datetime'].iloc[1:],
            y=z_returns_secondary,
            mode='lines+markers',
            name=secondary_pair
        ))
    fig_std.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_std.update_layout(
        title="Standardization of Daily Returns (Z-score)",
        xaxis_title="Date",
        yaxis_title="Z-score"
    )
    st.plotly_chart(fig_std, use_container_width=True)

    # =========================================================================
    # Graph 5: Frequency Distribution of Daily Returns (Dotted Line Graph)
    # =========================================================================
    st.subheader("5. Frequency Distribution of Daily Returns (%)")
    freq_primary, bins_primary = np.histogram(returns_primary, bins=10)
    bin_centers_primary = 0.5 * (bins_primary[:-1] + bins_primary[1:])
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(
        x=bin_centers_primary,
        y=freq_primary,
        mode='lines+markers',
        line=dict(dash='dot'),
        name=primary_pair
    ))
    if df_secondary is not None:
        freq_secondary, bins_secondary = np.histogram(returns_secondary, bins=10)
        bin_centers_secondary = 0.5 * (bins_secondary[:-1] + bins_secondary[1:])
        fig_freq.add_trace(go.Scatter(
            x=bin_centers_secondary,
            y=freq_secondary,
            mode='lines+markers',
            line=dict(dash='dot'),
            name=secondary_pair
        ))
    fig_freq.update_layout(
        title="Frequency Distribution of Daily Returns (%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # =========================================================================
    # Graph 6: Trend Analysis Pie Chart (Bullish vs Bearish Days)
    # =========================================================================
    st.subheader("6. Trend Analysis: Bullish vs Bearish Days")
    diff_primary = df_primary['close'].diff().dropna()
    bullish_count = (diff_primary > 0).sum()
    bearish_count = (diff_primary < 0).sum()
    labels = ["Bullish Days", "Bearish Days"]
    values = [bullish_count, bearish_count]
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig_pie.update_layout(title="Bullish vs Bearish Days")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Trading Recommendation based on bullish/bearish count and trend slope
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(df_primary)), df_primary['close'])
    signal_from_days = "Long" if bullish_count > bearish_count else "Short" if bearish_count > bullish_count else "Neutral"
    signal_from_trend = "Long" if slope > 0 else "Short" if slope < 0 else "Neutral"
    if signal_from_days == signal_from_trend:
        recommendation = signal_from_days
    else:
        recommendation = "Mixed signals - further analysis needed"
    st.markdown(f"### Trading Recommendation: **{recommendation}**")

    # =========================================================================
    # Graph 7: OHLCV Candlestick Chart with Volume
    # =========================================================================
    st.subheader("7. OHLCV Candlestick Chart (Last 30 Days)")
    fig_ohlc = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig_ohlc.add_trace(go.Candlestick(
        x=df_primary['datetime'],
        open=df_primary['open'],
        high=df_primary['high'],
        low=df_primary['low'],
        close=df_primary['close'],
        name='Price'
    ), row=1, col=1)
    fig_ohlc.add_trace(go.Bar(
        x=df_primary['datetime'],
        y=df_primary['volume'],
        name='Volume'
    ), row=2, col=1)
    fig_ohlc.update_layout(
        title=f"OHLCV Chart for {primary_pair}",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig_ohlc, use_container_width=True)

# =============================================================================
# Cync Page (Unchanged)
# =============================================================================

def cync_page():
    # Auto-refresh every 5 minutes (300,000 milliseconds)
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="data_refresh")
    
    st.title("Cync - 5-Minute Candle Price Changes")
    st.write("Select a crypto asset to view price changes across multiple pairs.")
    
    # Timezone selector
    tz = st.selectbox("Select Time Zone", ["UTC", "IST"], index=0)

    available_assets = ["BNB", "XRP", "ADA", "EOS", "NEO", "XLM"]
    selected_asset = st.selectbox("Select Crypto Asset", available_assets, index=0)
    quote_currencies = ["BTC", "ETH", "USDT"]
    pairs = [f"{selected_asset}/{quote}" for quote in quote_currencies]

    # For each pair, fetch the last 6 candles (6 candles give us 5 consecutive percentage changes)
    limit = 6
    data = {}  # Dictionary mapping pair -> DataFrame of the last `limit` candles
    for pair in pairs:
        df = fetch_ohlcv(pair, '5m', since=None, limit=limit)
        if df is None or df.empty or len(df) < 2:
            st.warning(f"Not enough data for {pair}")
        else:
            data[pair] = df

    if not data:
        st.error("No data available for any pair.")
        return

    # Loop from the latest candle (index limit-1) to the second candle (index 1)
    # so that the latest graph appears first.
    for i in range(limit - 1, 0, -1):
        price_changes = {}
        candle_time = None
        for pair, df in data.items():
            if len(df) <= i:
                continue
            last_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-1]
            change_percent = ((last_close - prev_close) / prev_close) * 100
            price_changes[pair] = change_percent
            if candle_time is None:
                candle_time = df['datetime'].iloc[i]
        
        if price_changes and candle_time is not None:
            if tz == "IST":
                local_time = candle_time + datetime.timedelta(hours=5, minutes=30)
                time_label = local_time.strftime('%H:%M') + " IST"
            else:
                time_label = candle_time.strftime('%H:%M') + " UTC"
            
            st.write(f"### Price Changes for {time_label}")
            fig = go.Figure()
            colors = ['green' if v > 0 else 'red' for v in price_changes.values()]
            fig.add_trace(go.Bar(
                x=list(price_changes.keys()),
                y=list(price_changes.values()),
                marker_color=colors
            ))
            fig.update_layout(
                title=f"Percentage Price Change at {time_label}",
                xaxis_title="Pair",
                yaxis_title="% Change"
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Calculator Page (Unchanged)
# =============================================================================

def calculator_page():
    st.title("Trading Calculator")
    st.write("Use the calculator below to estimate your risk and reward for a trade.")
    
    with st.form(key='calculator_form'):
        capital = st.number_input("Capital (USD)", min_value=100.0, value=10000.0, step=100.0)
        entry_price = st.number_input("Entry Price", min_value=0.001, value=1.0, step=0.001)
        tp_percent = st.number_input("Take Profit (%)", min_value=0.1, value=1.0, step=0.1)
        sl_percent = st.number_input("Stop Loss (%)", min_value=0.1, value=0.5, step=0.1)
        risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, value=2.0, step=0.1)
        submit_button = st.form_submit_button(label='Calculate')
    
    if submit_button:
        risk_amount = capital * (risk_percent / 100.0)
        sl_price = entry_price * (1 - sl_percent / 100.0)
        tp_price = entry_price * (1 + tp_percent / 100.0)
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            st.error("Stop Loss cannot be equal to Entry Price.")
            return
        position_size = risk_amount / risk_per_unit
        potential_profit = tp_price - entry_price
        potential_loss = entry_price - sl_price
        total_profit = potential_profit * position_size
        total_loss = potential_loss * position_size
        
        st.subheader("Calculation Results")
        st.write(f"Risk Amount: **${risk_amount:.2f}**")
        st.write(f"Position Size (units): **{position_size:.4f}**")
        st.write(f"Stop Loss Price: **${sl_price:.4f}**")
        st.write(f"Take Profit Price: **${tp_price:.4f}**")
        st.write(f"Potential Profit per Unit: **${potential_profit:.4f}**")
        st.write(f"Potential Loss per Unit: **${potential_loss:.4f}**")
        st.write(f"Total Potential Profit: **${total_profit:.2f}**")
        st.write(f"Total Potential Loss: **${total_loss:.2f}**")

# =============================================================================
# Main Navigation
# =============================================================================

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Dashboard", "Cync", "Calculator"))
    
    if page == "Dashboard":
        dashboard_page()
    elif page == "Cync":
        cync_page()
    elif page == "Calculator":
        calculator_page()

if __name__ == '__main__':
    main()
