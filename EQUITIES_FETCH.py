import pandas as pd
import yfinance as yf
import time
import json
import os
from datetime import datetime

def fetch_stock_data(json_file_path, start_date, end_date, delay=0.5):
    # Load the JSON file
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    with open(json_file_path, 'r') as file:
        equities_data = json.load(file)
    
    # Extract tickers and categories from the JSON
    all_close_prices = pd.DataFrame()
    tickers_dict = []
    for category, tickers in equities_data["SHORT_MEDIUM_EQUITIES_LIST"].items():
        for ticker in tickers:
            tickers_dict.append((ticker, category))
    
    # Fetch data for each ticker
    for ticker, category in tickers_dict:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if len(data) < 20:
                print(f"{ticker} ({category}): Insufficient data (<20 days)")
                continue
            # Create a DataFrame with Close, Ticker, and Category, keeping Date as index
            close_data = data['Close'].to_numpy().flatten()
            close_prices = pd.DataFrame({
                'Close': close_data,
                'Ticker': ticker,
                'Category': category
            }, index=data.index)
            all_close_prices = pd.concat([all_close_prices, close_prices], ignore_index=False)
            time.sleep(delay)  # Rate limiting
        except Exception as e:
            print(f"{ticker} ({category}): Error fetching data - {e}")
            continue
    return all_close_prices

def main():
    # Configuration
    start_date = '2024-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')  # Use today's date to avoid future dates

    # Load tickers from JSON
    json_file_path = 'equities.json'
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found. Please ensure the file exists.")
        return
    
    with open(json_file_path, 'r') as file:
        tickers_data = json.load(file)
    
    # Combine all tickers into a single list
    all_tickers = [(ticker, category) for category, tick_list in tickers_data["EQUITIES_LIST"].items() for ticker in tick_list]

    # Fetch stock closing prices
    stock_data = fetch_stock_data(json_file_path, start_date, end_date)

    if not stock_data.empty:
        # Ensure the index is named 'Date' for clarity
        stock_data.index.name = 'Date'
        # Save to CSV
        stock_data.to_csv('all_stock_data.csv')
        print(f"\nCombined closing price data for tickers saved to 'all_stock_data.csv'")
    else:
        print("\nNo valid stock data to combine.")

if __name__ == "__main__":
    main()