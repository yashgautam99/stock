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

def plot_macd(data):
    """
    Plots the MACD and Signal line.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data.
    """
    # Calculate the MACD and Signal line
    shortEMA = data['Close'].ewm(span=12, adjust=False).mean()
    longEMA = data['Close'].ewm(span=26, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=9, adjust=False).mean()

    # Plot MACD and Signal line
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, MACD, label='MACD', color='r', linewidth=1)
    ax.plot(data.index, signal, label='Signal line', color='b', linewidth=1)
    ax.set_title('MACD Analysis')
    ax.legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_price_changes(data):
    """
    Plots Absolute and Percentage Price Changes.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data.
    """
    data['Absolute_Change'] = data['Close'].diff()
    data['Percentage_Change'] = data['Close'].pct_change() * 100

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(data.index, data['Absolute_Change'], label='Absolute Price Change', color='blue', linewidth=0.5)
    axs[0].set_title('Absolute Price Change')
    axs[0].set_ylabel('Price Change')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(data.index, data['Percentage_Change'], label='Percentage Price Change', color='red', linewidth=0.5)
    axs[1].set_title('Percentage Price Change')
    axs[1].set_ylabel('Percentage Change')
    axs[1].set_xlabel('Date')
    axs[1].legend()
    axs[1].grid(True)

    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_price_range(data):
    """
    Plots the daily price range, calculated as the difference between the high and low prices.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data.
    """
    data['Daily_Range'] = data['High'] - data['Low']  # Calculate the daily price range

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(data.index, data['Daily_Range'], label='Daily Price Range', linewidth=0.5)
    ax.set_title('Daily Price Range')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price Range')
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
            st.subheader(f'{stock} Stock Price')
            st.write("""
                This graph shows the closing price of the selected stock over time. The closing price is the final price at which the stock traded on a given day, providing a snapshot of its value.
            """)
            plot_stock_data(stock_data, f'{stock} Stock Price')

            # Plot additional data
            st.subheader('Additional Price and Volume Data')
            st.write("""
                This section includes:
                - **Open Price**: The price at which the stock started trading on a given day.
                - **High and Low Prices**: The highest and lowest prices the stock reached during the trading day.
                - **Volume**: The number of shares traded during the day, indicating market activity and interest.
            """)
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
                ["Select", "Moving Averages", "Bollinger Bands", "RSI", "MACD", "Price Changes", "Daily Price Range"]
            )

            # Plot selected indicator
            if indicator == "Moving Averages":
                st.subheader('Moving Averages')
                st.write("""
                    The 20-day and 50-day Simple Moving Averages (SMA) are plotted alongside the closing price. These lines smooth out price fluctuations, making it easier to identify trends. A rising SMA indicates an upward trend, while a falling SMA suggests a downward trend.
                """)
                plot_moving_averages(stock_data)
            elif indicator == "Bollinger Bands":
                st.subheader('Bollinger Bands')
                st.write("""
                    Bollinger Bands consist of a middle band (20-day SMA) and two outer bands (upper and lower bands). The bands expand during high volatility and contract during low volatility. They help identify overbought and oversold conditions and potential price reversals.
                """)
                plot_bollinger_bands(stock_data)
            elif indicator == "RSI":
                st.subheader('Relative Strength Index (RSI)')
                st.write("""
                    The RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100, with values above 70 indicating overbought conditions and below 30 indicating oversold conditions. It's useful for identifying potential reversals.
                """)
                plot_rsi(stock_data)
            elif indicator == "MACD":
                st.subheader('MACD (Moving Average Convergence Divergence)')
                st.write("""
                    The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a stock's price. The MACD line is the difference between the 12-day and 26-day EMAs, while the signal line is a 9-day EMA of the MACD. Crosses of the MACD and signal line can indicate buy or sell signals.
                """)
                plot_macd(stock_data)
            elif indicator == "Price Changes":
                st.subheader('Price Changes')
                st.write("""
                    This plot shows the absolute and percentage changes in the stock's closing price. Large changes may indicate significant market events or trends. The absolute change shows the raw price difference, while the percentage change provides a relative measure.
                """)
                plot_price_changes(stock_data)
            elif indicator == "Daily Price Range":
                st.subheader('Daily Price Range')
                st.write("""
                    The daily price range shows the difference between the highest and lowest prices of the stock on a given day. This metric helps understand the volatility of the stock during the trading day.
                """)
                plot_price_range(stock_data)
            
        else:
            st.write("No stock data available. Please select a stock from the list.")

if __name__ == "__main__":
    main()