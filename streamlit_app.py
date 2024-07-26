import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import ta

# List of 50 stock tickers for user selection
STOCKS = ['Select','AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'NVDA', 'JNJ', 'V',
          'WMT', 'JPM', 'PG', 'UNH', 'MA', 'DIS', 'HD', 'PYPL', 'VZ', 'ADBE', 'NFLX',
          'INTC', 'PFE', 'KO', 'PEP', 'NKE', 'MRK', 'T', 'BA', 'CSCO', 'ABT', 'XOM',
          'CRM', 'ACN', 'CMCSA', 'AVGO', 'MCD', 'QCOM', 'MDT', 'HON', 'COST', 'AMGN',
          'TMUS', 'TXN', 'NEE', 'PM', 'IBM', 'LMT', 'ORCL', 'INTU']

def load_stock_data(ticker, start, end):
    """
    Fetches historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start (datetime): Start date for data retrieval.
    end (datetime): End date for data retrieval.

    Returns:
    pd.DataFrame: DataFrame containing stock data or empty DataFrame on error.
    """
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def plot_stock_data(data, title):
    """
    Plots the closing price of the stock.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data.
    title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_additional_data(data):
    """
    Plots additional stock data including Open, High, Low prices, and Volume.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Plot Close and Open prices
    axs[0].plot(data.index, data['Close'], label='Close Price', color='blue')
    axs[0].plot(data.index, data['Open'], label='Open Price', color='green')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].set_title('Stock Prices')

    # Plot High and Low prices
    axs[1].plot(data.index, data['High'], label='High Price', color='red')
    axs[1].plot(data.index, data['Low'], label='Low Price', color='orange')
    axs[1].set_ylabel('Price')
    axs[1].legend()

    # Plot Volume
    axs[2].plot(data.index, data['Volume'], label='Volume', color='purple')
    axs[2].set_ylabel('Volume')
    axs[2].set_xlabel('Date')
    axs[2].legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_moving_averages(data):
    """
    Plots moving averages (20-day and 50-day) along with the closing price.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data with calculated moving averages.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data.index, data['Close'], label='Close Price', linewidth=0.5)
    ax.plot(data.index, data['SMA_20'], label='20-Day SMA', linestyle='--')
    ax.plot(data.index, data['SMA_50'], label='50-Day SMA', linestyle='--')

    ax.set_title('Moving Average Analysis')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_bollinger_bands(data):
    """
    Plots Bollinger Bands along with the closing price.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data with calculated Bollinger Bands.
    """
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
    """
    Plots the Relative Strength Index (RSI) along with overbought and oversold lines.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data with calculated RSI.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data.index, data['RSI'], label='RSI', linewidth=0.5)
    ax.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought')
    ax.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold')
    ax.set_title('RSI Analysis')
    ax.legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("📊 Stock Prediction Dashboard")

    # Sidebar for user inputs
    st.sidebar.header("Select Stock and Date Range")
    stock = st.sidebar.selectbox("Choose a stock", STOCKS)
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    # Validate date range
    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
    else:
        # Load stock data
        stock_data = load_stock_data(stock, start_date, end_date)
        
        if not stock_data.empty:
            # Display stock data
            st.subheader(f"{stock} Stock Data")
            st.write(stock_data.tail())

            # Plot the close price
            plot_stock_data(stock_data, f'{stock} Stock Price')
            
            # Plot additional data
            plot_additional_data(stock_data)
            
            # Calculate moving averages
            stock_data['SMA_20'] = stock_data['Close'].rolling(20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(50).mean()

            # Calculate Bollinger Bands
            bb = ta.volatility.BollingerBands(stock_data['Close'], window=20, window_dev=2)
            stock_data['BB_High'] = bb.bollinger_hband()
            stock_data['BB_Low'] = bb.bollinger_lband()

            # Calculate RSI
            stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()

            # Sidebar for selecting additional indicators
            indicator = st.sidebar.selectbox(
                "Choose indicators",
                ["Select", "Moving Averages", "Bollinger Bands", "RSI"]
            )

            # Plot selected indicator
            if indicator == "Moving Averages":
                plot_moving_averages(stock_data)
            elif indicator == "Bollinger Bands":
                plot_bollinger_bands(stock_data)
            elif indicator == "RSI":
                plot_rsi(stock_data)
            
        else:
            st.write("No stock data available. Please select a stock from the list.")

if __name__ == "__main__":
    main()
