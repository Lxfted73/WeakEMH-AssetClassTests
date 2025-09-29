import pandas as pd
import yfinance as yf
import time
import json
import os
from datetime import datetime

def fetch_stock_data(json_file_path, start_date, end_date, delay=0.5):
    """
    Fetch stock price data for tickers specified in a JSON file using yfinance.
    
    Parameters:
    - json_file_path: Path to JSON file containing tickers and categories
    - start_date: Start date for data retrieval (YYYY-MM-DD)
    - end_date: End date for data retrieval (YYYY-MM-DD)
    - delay: Delay between API calls to avoid rate limiting (seconds)
    
    Returns:
    - DataFrame with columns ['Close', 'Ticker', 'Category'] and Date as index
    """
    # Load the JSON file
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    with open(json_file_path, 'r') as file:
        equities_data = json.load(file)
    
    # Extract tickers and categories from the JSON
    all_close_prices = pd.DataFrame()
    tickers_dict = []
    for category, tickers in equities_data["MEDIUM_EQUITIES_LIST"].items():
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
    """
    Main function to fetch stock data using dates from date_config.json and save to CSV.
    """
    # Load date configuration
    date_config_path = 'date_config.json'
    try:
        with open(date_config_path, 'r') as file:
            date_config = json.load(file)
        start_date = date_config['START_DATE']
        end_date = date_config['END_DATE']
        print(f"Using date range: {start_date} to {end_date}")
    except FileNotFoundError:
        print(f"Error: {date_config_path} not found. Please ensure the file exists.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in {date_config_path}.")
        return
    
    # Load tickers from JSON
    json_file_path = 'equities.json'
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found. Please ensure the file exists.")
        return
    
    with open(json_file_path, 'r') as file:
        tickers_data = json.load(file)
    
    # Combine all tickers into a single list
    all_tickers = [(ticker, category) for category, tick_list in tickers_data["EQUITIES_LIST"].items() for ticker in tick_list]
    print(f"Found {len(all_tickers)} tickers to process")

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