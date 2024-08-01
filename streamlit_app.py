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
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
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

# Function to plot Price Range
def plot_price_range(data):
    data['Range'] = data['High'] - data['Low']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Range'], mode='lines', name='Price Range', line=dict(color='blue')))
    fig.update_layout(title='Price Range', xaxis_title='Date', yaxis_title='Range')
    st.plotly_chart(fig)

# Function to build and train LSTM model
@st.cache_data
def build_and_train_lstm(data, window_size):
    # Prepare the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])
    
    X, y = np.array(X), np.array(y)
    
    # Build the model
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    return model, scaler, data

# Function to forecast stock prices
@st.cache_data
def forecast_stock(_scaler, model, data, window_size, future_steps):
    scaler = _scaler
    scaled_data = scaler.transform(data[['Close']])
    
    # Prepare the initial input for prediction
    X_input = scaled_data[-window_size:].reshape((1, window_size, 1))
    
    forecast = []
    for _ in range(future_steps):
        pred = model.predict(X_input)[0][0]
        forecast.append(pred)
        
        # Prepare the new input for the next prediction
        # Create a new array with the predicted value
        new_input = np.array([[pred]]).reshape((1, 1, 1))
        
        # Concatenate the new input to maintain the window size
        X_input = np.concatenate((X_input[:, 1:, :], new_input), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, future_steps + 1)]
    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])
    return forecast_df

# Function to plot the forecast
def plot_forecast(data, forecast, window_size):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
    fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Main function to run the Streamlit app
def main():
    st.title("Stock Prediction Dashboard")

    selected_stock = st.selectbox("Select a stock:", STOCKS)

    if selected_stock != 'Select':
        start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
        end_date = st.date_input("End Date", datetime.date.today())
        window_size = st.slider("Window Size for LSTM", min_value=1, max_value=60, value=20)
        future_steps = st.slider("Days to Forecast", min_value=1, max_value=365, value=30)

        data = load_stock_data(selected_stock, start_date, end_date)

        if not data.empty:
            st.subheader(f"{selected_stock} Stock Data")
            st.write(data.head())

            # Plot the stock data
            plot_stock_data(data, f"{selected_stock} Stock Price")

            # Compute and plot technical indicators
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            plot_moving_averages(data)

            data['BB_High'], data['BB_Low'], data['BB_Mid'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband(), ta.volatility.BollingerBands(data['Close']).bollinger_lband(), ta.volatility.BollingerBands(data['Close']).bollinger_mavg()
            plot_bollinger_bands(data)

            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            plot_rsi(data)

            plot_macd(data)
            plot_price_changes(data)
            plot_price_range(data)

            # Train the LSTM model and make forecasts
            model, scaler, history = build_and_train_lstm(data, window_size)
            forecast = forecast_stock(scaler, model, data, window_size, future_steps)
            plot_forecast(data, forecast, window_size)

            # Display model performance metrics
            y_true = data['Close'].values[-len(forecast):]
            st.write(f"Mean Squared Error: {mean_squared_error(y_true, forecast['Forecast'].values.flatten()):.4f}")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_true, forecast['Forecast'].values.flatten()):.4f}")

if __name__ == "__main__":
    main()
