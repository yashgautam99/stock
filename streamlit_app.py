import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import ta

# List of 50 stock tickers
STOCKS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'NVDA', 'JNJ', 'V',
          'WMT', 'JPM', 'PG', 'UNH', 'MA', 'DIS', 'HD', 'PYPL', 'VZ', 'ADBE', 'NFLX',
          'INTC', 'PFE', 'KO', 'PEP', 'NKE', 'MRK', 'T', 'BA', 'CSCO', 'ABT', 'XOM',
          'CRM', 'ACN', 'CMCSA', 'AVGO', 'MCD', 'QCOM', 'MDT', 'HON', 'COST', 'AMGN',
          'TMUS', 'TXN', 'NEE', 'PM', 'IBM', 'LMT', 'ORCL', 'INTU']

def load_stock_data(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def plot_stock_data(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_additional_data(data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    axs[0].plot(data.index, data['Close'], label='Close Price', color='blue')
    axs[0].plot(data.index, data['Open'], label='Open Price', color='green')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].set_title('Stock Prices')

    axs[1].plot(data.index, data['High'], label='High Price', color='red')
    axs[1].plot(data.index, data['Low'], label='Low Price', color='orange')
    axs[1].set_ylabel('Price')
    axs[1].legend()

    axs[2].plot(data.index, data['Volume'], label='Volume', color='purple')
    axs[2].set_ylabel('Volume')
    axs[2].set_xlabel('Date')
    axs[2].legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_moving_averages(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data.index, data['Close'], label='Close Price', linewidth=0.5)
    ax.plot(data.index, data['SMA_20'], label='20-Day SMA', linestyle='--')
    ax.plot(data.index, data['SMA_50'], label='50-Day SMA', linestyle='--')

    ax.set_title('Moving Average Analysis')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_bollinger_bands(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data.index, data['Close'], label='Close Price', linewidth=0.5)
    ax.plot(data.index, data['BB_High'], label='Bollinger Band High', linestyle='--')
    ax.plot(data.index, data['BB_Low'], label='Bollinger Band Low', linestyle='--')
    ax.fill_between(data.index, data['BB_High'], data['BB_Low'], color='gray', alpha=0.2)

    ax.set_title('Bollinger Bands Analysis')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_rsi(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data.index, data['RSI'], label='RSI', linewidth=0.5)
    ax.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought')
    ax.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold')
    ax.set_title('RSI Analysis')
    ax.legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    st.title("🎈 Stock Prediction Dashboard")

    # Dropdown menu for stock selection
    st.sidebar.header("Select Stock and Date Range")
    stock = st.sidebar.selectbox("Choose a stock", STOCKS)
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
    else:
        stock_data = load_stock_data(stock, start_date, end_date)
        
        if not stock_data.empty:
            st.subheader(f"{stock} Stock Data")
            st.write(stock_data.tail())

            # Plot the close price
            plot_stock_data(stock_data, f'{stock} Stock Price')
            
            # Plot additional data
            plot_additional_data(stock_data)
            
            # Calculate moving averages
            stock_data['SMA_20'] = stock_data['Close'].rolling(20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(50).mean()
            
            # Plot moving averages
            plot_moving_averages(stock_data)

            # Calculate Bollinger Bands
            bb = ta.volatility.BollingerBands(stock_data['Close'], window=20, window_dev=2)
            stock_data['BB_High'] = bb.bollinger_hband()
            stock_data['BB_Low'] = bb.bollinger_lband()

            # Plot Bollinger Bands
            plot_bollinger_bands(stock_data)
            
            # Calculate RSI
            stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()

            # Plot RSI
            plot_rsi(stock_data)
            
        else:
            st.write("No data available for the selected stock and date range.")

if __name__ == "__main__":
    main()
