import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import plotly.graph_objects as go

# List of 50 stock tickers for user selection
STOCKS = ['Select', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK-B', 'NVDA', 'JNJ', 'V',
          'WMT', 'JPM', 'PG', 'UNH', 'MA', 'DIS', 'HD', 'PYPL', 'VZ', 'ADBE', 'NFLX',
          'INTC', 'PFE', 'KO', 'PEP', 'NKE', 'MRK', 'T', 'BA', 'CSCO', 'ABT', 'XOM',
          'CRM', 'ACN', 'CMCSA', 'AVGO', 'MCD', 'QCOM', 'MDT', 'HON', 'COST', 'AMGN',
          'TMUS', 'TXN', 'NEE', 'PM', 'IBM', 'LMT', 'ORCL', 'INTU']

# Function to load stock data
def load_stock_data(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        stock_data = stock_data.dropna()
        return stock_data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

# Function to plot stock data with Plotly for interactivity
def plot_stock_data(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig)

# Function to plot additional data with Plotly
def plot_additional_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume'], mode='lines', name='Volume', line=dict(color='purple')))
    fig.update_layout(title='Stock Prices and Volume', xaxis_title='Date', yaxis_title='Price / Volume')
    st.plotly_chart(fig)

# Function to plot moving averages
def plot_moving_averages(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='20-Day SMA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(dash='dash')))
    fig.update_layout(title='Moving Average Analysis', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Function to plot Bollinger Bands
def plot_bollinger_bands(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Bollinger Band High', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Bollinger Band Low', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Mid'], mode='lines', name='Bollinger Band Mid', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='Bollinger Bands Analysis', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Function to plot RSI
def plot_rsi(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='blue')))
    fig.add_shape(type='line', x0=data.index[0], x1=data.index[-1], y0=70, y1=70, line=dict(color='red', dash='dash'))
    fig.add_shape(type='line', x0=data.index[0], x1=data.index[-1], y0=30, y1=30, line=dict(color='green', dash='dash'))
    fig.update_layout(title='RSI Analysis', xaxis_title='Date', yaxis_title='RSI')
    st.plotly_chart(fig)

# Function to plot MACD
def plot_macd(data):
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=9, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=MACD, mode='lines', name='MACD', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=signal, mode='lines', name='Signal Line', line=dict(color='blue')))
    fig.update_layout(title='MACD Analysis', xaxis_title='Date', yaxis_title='MACD')
    st.plotly_chart(fig)

# Function to plot Price Changes
def plot_price_changes(data):
    data['Absolute_Change'] = data['Close'].diff()
    data['Percentage_Change'] = data['Close'].pct_change() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Absolute_Change'], mode='lines', name='Absolute Price Change', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Percentage_Change'], mode='lines', name='Percentage Price Change', line=dict(color='red')))
    fig.update_layout(title='Price Changes', xaxis_title='Date', yaxis_title='Change')
    st.plotly_chart(fig)

# Function to plot Daily Price Range
def plot_price_range(data):
    data['Daily_Range'] = data['High'] - data['Low']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Daily_Range'], mode='lines', name='Daily Price Range', line=dict(color='blue')))
    fig.update_layout(title='Daily Price Range', xaxis_title='Date', yaxis_title='Price Range')
    st.plotly_chart(fig)

# Function to build and train the LSTM model
@st.cache_data
def build_and_train_lstm(data, window_size):
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    # Preparing the data for LSTM
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Splitting the data into training and validation sets
    split = int(X.shape[0] * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Building the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    return model, scaler, history

# Function to forecast stock prices
@st.cache_data
def forecast_stock(_scaler, model, data, window_size, future_steps):
    # Scaling the data
    scaled_data = _scaler.transform(data[['Close']])

    # Preparing the last window of data for prediction
    last_sequence = scaled_data[-window_size:]
    predictions = []

    for _ in range(future_steps):
        next_prediction = model.predict(last_sequence.reshape(1, window_size, 1))
        predictions.append(next_prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction)

    # Inverse transform to get original scale
    predictions = _scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def main():
    st.title("📈 Stock Prediction Dashboard")

    st.sidebar.header("Select Stock and Time Period")
    stock = st.sidebar.selectbox("Choose a stock", STOCKS)
    interval = st.sidebar.selectbox("Data Interval", ['1d', '1wk', '1mo'])
    start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())
    future_steps = st.sidebar.slider("Days to Predict", 1, 30, 7)
    window_size = st.sidebar.slider("Window Size for LSTM", 30, 120, 60)

    if stock == 'Select':
        st.sidebar.error("Please select a stock.")
    elif start_date >= end_date:
        st.sidebar.error("End date must be after start date.")
    else:
        stock_data = load_stock_data(stock, start_date, end_date)
        if stock_data.empty:
            st.write("No stock data available. Please select a different stock.")
            return

        st.subheader(f"{stock} Stock Data")
        st.dataframe(stock_data.tail())

        st.subheader(f'{stock} Stock Price')
        plot_stock_data(stock_data, f'{stock} Stock Price')

        st.subheader('Additional Stock Data')
        plot_additional_data(stock_data)

        stock_data['SMA_20'] = ta.trend.sma_indicator(stock_data['Close'], window=20)
        stock_data['SMA_50'] = ta.trend.sma_indicator(stock_data['Close'], window=50)
        st.subheader('Moving Averages')
        plot_moving_averages(stock_data)

        stock_data['BB_High'] = ta.volatility.bollinger_hband(stock_data['Close'])
        stock_data['BB_Low'] = ta.volatility.bollinger_lband(stock_data['Close'])
        stock_data['BB_Mid'] = ta.volatility.bollinger_mavg(stock_data['Close'])
        st.subheader('Bollinger Bands')
        plot_bollinger_bands(stock_data)

        stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14)
        st.subheader('RSI (Relative Strength Index)')
        plot_rsi(stock_data)

        st.subheader('MACD (Moving Average Convergence Divergence)')
        plot_macd(stock_data)

        st.subheader('Price Changes')
        plot_price_changes(stock_data)

        st.subheader('Daily Price Range')
        plot_price_range(stock_data)

        model, scaler, history = build_and_train_lstm(stock_data, window_size)
        predictions = forecast_stock(scaler, model, stock_data, window_size, future_steps)

        st.subheader('Forecasted Stock Prices')
        future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig.update_layout(title=f'{stock} Forecasted Stock Prices', xaxis_title='Date', yaxis_title='Stock Price')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
