import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks

# --- Functions ---
@st.cache_data
def download(symbol, interval, period='1mo', start_date=None, end_date=None):
    data = yf.download(tickers=symbol, interval=interval, period=period, start=start_date, end=end_date)
    return data

@st.cache_data
def calculate_rsi_wilder(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def identify_divergences(close_prices, rsi_values, window=5, prominence=2):
    price_peaks, _ = find_peaks(close_prices, distance=window, prominence=prominence)
    price_troughs, _ = find_peaks(-np.array(close_prices), distance=window, prominence=prominence)
    rsi_peaks, _ = find_peaks(rsi_values, distance=window, prominence=prominence)
    rsi_troughs, _ = find_peaks(-np.array(rsi_values), distance=window, prominence=prominence)
    bullish_div = []
    bearish_div = []
    for i in range(1, len(price_troughs)):
        price_idx1, price_idx2 = price_troughs[i - 1], price_troughs[i]
        if close_prices[price_idx2] < close_prices[price_idx1]:
            rsi_trough_idx1 = find_closest_index(rsi_troughs, price_idx1)
            rsi_trough_idx2 = find_closest_index(rsi_troughs, price_idx2)
            if rsi_trough_idx1 is not None and rsi_trough_idx2 is not None:
                idx1, idx2 = rsi_troughs[rsi_trough_idx1], rsi_troughs[rsi_trough_idx2]
                if rsi_values[idx2] > rsi_values[idx1]:
                    bullish_div.append((price_idx2, idx2))
    for i in range(1, len(price_peaks)):
        price_idx1, price_idx2 = price_peaks[i - 1], price_peaks[i]
        if close_prices[price_idx2] > close_prices[price_idx1]:
            rsi_peak_idx1 = find_closest_index(rsi_peaks, price_idx1)
            rsi_peak_idx2 = find_closest_index(rsi_peaks, price_idx2)
            if rsi_peak_idx1 is not None and rsi_peak_idx2 is not None:
                idx1, idx2 = rsi_peaks[rsi_peak_idx1], rsi_peaks[rsi_peak_idx2]
                if rsi_values[idx2] < rsi_values[idx1]:
                    bearish_div.append((price_idx2, idx2))
    return {'bullish': bullish_div, 'bearish': bearish_div}

@st.cache_data
def find_closest_index(indices, target_idx, max_distance=200):
    if len(indices) == 0:
        return None
    distances = np.abs(indices - target_idx)
    min_idx = np.argmin(distances)
    if distances[min_idx] <= max_distance:
        return min_idx
    return None

def main(tickers, interval, period='1mo', start_date=None, end_date=None):
    all_divergences = {}
    close_prices = []
    rsi_values = []
    progress_bar = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            ticker_data = download(symbol=ticker + '.NS', interval=interval, period=period, start_date=start_date, end_date=end_date)
            close = ticker_data['Close']
            close_prices.append(close)
            close_cleaned = close.values.ravel()
            rsi = calculate_rsi_wilder(close, 14)
            rsi_cleaned = rsi.values.ravel()
            rsi_values.append(rsi_cleaned)
            divergences = identify_divergences(close_cleaned, rsi_cleaned)
            all_divergences[ticker] = divergences
            progress_bar.progress((i + 1) / len(tickers))
        except Exception as e:
            st.error(f'Error processing {ticker}: {e}')
    return all_divergences, close_prices, rsi_values

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title('RSI Divergence Screener')

# Inputs
uploaded_file = st.file_uploader("Upload CSV with Symbols (for Ticker name only, data will be downloaded by Yfinance function)", type=["csv"])
col1, col2 = st.columns(2)
with col1:
    INTERVAL = st.selectbox('Interval', ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'], index=8)
with col2:
    PERIOD = st.selectbox('Period', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=1) #corrected index

col3, col4 = st.columns(2)
with col3:
    START_DATE = st.date_input('Start Date', value=None)
with col4:
    END_DATE = st.date_input('End Date', value=None)

if st.button('Run Analysis'):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, usecols=['Symbol'])
            if 'Symbol' not in df.columns:
                st.error("CSV must contain a 'Symbol' column.")
                st.stop()
            TICKERS = df['Symbol'].tolist()
            divergence_values, closes, rsi_values = main(TICKERS, INTERVAL, PERIOD, START_DATE, END_DATE)
            screened = []
            for tickers, divergences in divergence_values.items():
                if len(divergences['bullish']) and len(divergences['bearish']) == 0:
                    screened.append(tickers)
            st.subheader('Screened Tickers (Bullish Divergence Only):')
            st.write(screened)
            st.write(df.info())
            del TICKERS, divergence_values, closes, rsi_values #clear memory
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a CSV file with symbols.")
