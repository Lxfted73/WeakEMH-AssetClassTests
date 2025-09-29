import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# List of input CSV files (add or remove files to include additional categories)
INPUT_FILES = ['data_all_equities.csv', 'data_random_walk.csv']

def ljung_box_test(all_stock_data, num_tickers_to_run='all'):
    """
    Perform a Ljung-Box test on stock price data to assess randomness of returns, organized by category.
    
    Parameters:
    - all_stock_data: DataFrame with columns ['Category', 'Ticker', 'Date', 'Close']
    - num_tickers_to_run: 'all' or an integer to limit the number of tickers processed per category
    
    Returns:
    - Dictionary with categories as keys and lists of [ticker, lb_stats, p_values, randomness] as values
      where lb_stats and p_values are dictionaries with lags as keys
    """
    # Validate input data
    required_columns = ['Category', 'Ticker', 'Date', 'Close']
    if not all(col in all_stock_data.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
    
    # Define lags for Ljung-Box test
    lags = [1, 3, 5, 7, 10, 15, 20, 30, 45, 60]
    
    # Create a dictionary of tickers grouped by category and ticker
    grouped_data = {
        (category, ticker): group_df
        for (category, ticker), group_df in all_stock_data.groupby(['Category', 'Ticker'])
    }
    
    # Store results as a dictionary by category
    results_by_category = {}
    
    # Track unique categories to organize output
    categories = sorted(set(category for category, _ in grouped_data.keys()))  # Sort for consistent output
    
    # Process each category
    for category in categories:
        print(f"\n=== {category} Analysis ===")
        category_results = []
        
        # Get tickers for this category
        category_tickers = sorted([ticker for cat, ticker in grouped_data.keys() if cat == category])
        
        # Apply ticker limit if specified
        if num_tickers_to_run != 'all':
            try:
                num_tickers_to_run = int(num_tickers_to_run)
                category_tickers = category_tickers[:num_tickers_to_run]
            except ValueError:
                print(f"Invalid num_tickers_to_run value: {num_tickers_to_run}. Using all tickers.")
                num_tickers_to_run = 'all'
        
        for ticker in category_tickers:
            try:
                # Get data for the current ticker
                stock_data = grouped_data[(category, ticker)]
                
                if len(stock_data) < 20:
                    print(f"{ticker}: Insufficient data in CSV (<20 days)")
                    continue
                
                # Ensure Date is datetime and set as index
                stock_data = stock_data.copy()
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data = stock_data.set_index('Date')
                
                # Calculate returns
                prices = stock_data['Close']
                returns = prices.pct_change().dropna()  # Calculate percentage change
                
                # Perform Ljung-Box test for all lags
                lb_results = acorr_ljungbox(returns, lags=lags, return_df=True)
                lb_stats = {f'lag_{lag}': lb_results['lb_stat'].iloc[i] for i, lag in enumerate(lags)}
                p_values = {f'lag_{lag}': lb_results['lb_pvalue'].iloc[i] for i, lag in enumerate(lags)}
                
                # Determine randomness based on lag 60 (p-value < 0.05 indicates non-randomness)
                randomness = 'Non-random' if p_values['lag_60'] < 0.05 else 'Random'
                
                # Store results for display (use lag 60 for table)
                result_entry = [ticker, f"{lb_stats['lag_60']:.2f}", f"{p_values['lag_60']:.4f}", randomness]
                category_results.append(result_entry)
                
                # Add to results dictionary (store all lags)
                if category not in results_by_category:
                    results_by_category[category] = []
                results_by_category[category].append([ticker, lb_stats, p_values, randomness])
                
            except Exception as e:
                print(f"{ticker}: Error processing data - {e}")
                continue
        
        # Pretty print table for category (using lag 60)
        if category_results:
            print(tabulate(category_results,
                           headers=['Ticker', 'LB Statistic (Lag 60)', 'P-value (Lag 60)', 'Randomness'],
                           tablefmt='fancy_grid'))
        else:
            print(f"No valid data for tickers in category {category}")
    
    return results_by_category

def plot_ljung_box_results(results_by_category, output_dir='ljung_box_plots'):
    """
    Create a box plot for Ljung-Box test p-values (lag 60) across all categories using seaborn.
    
    Parameters:
    - results_by_category: Dictionary with categories as keys and lists of [ticker, lb_stats, p_values, randomness]
    - output_dir: Directory to save the plot image
    """
    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # # Commented out: Bar plot for Ljung-Box test statistics per category
    # for category, results in results_by_category.items():
    #     tickers = [result[0] for result in results]
    #     lb_stats = [float(result[1]['lag_60']) for result in results]  # Use lag 60
    #     
    #     plt.figure(figsize=(10, 6))
    #     sns.barplot(x=tickers, y=lb_stats, hue=tickers, palette='Blues', legend=False)
    #     plt.axhline(y=18.31, color='red', linestyle='--', label='Critical Value (95%, 10 lags)')
    #     plt.title(f'Ljung-Box Test Statistics (Lag 60) - {category}', fontsize=14)
    #     plt.xlabel('Ticker', fontsize=12)
    #     plt.ylabel('Ljung-Box Statistic', fontsize=12)
    #     plt.legend()
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     
    #     # Save the bar plot
    #     bar_filename = os.path.join(output_dir, f'ljung_box_bar_{category}.png')
    #     plt.savefig(bar_filename)
    #     plt.close()
    #     print(f"Bar plot for {category} saved as {bar_filename}")
    
    # Box plot for p-values across categories (using lag 60)
    p_values_by_category = []
    category_labels = []
    for category, results in results_by_category.items():
        p_values = [result[2]['lag_60'] for result in results]  # Use lag 60
        if p_values:
            p_values_by_category.append(p_values)
            category_labels.append(category)
    
    if p_values_by_category:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=p_values_by_category, palette='Blues')
        plt.xticks(ticks=range(len(category_labels)), labels=category_labels)
        plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')
        plt.title('P-value Distribution by Category (Lag 60)', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('P-value', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the box plot
        box_filename = os.path.join(output_dir, 'ljung_box_pvalue_boxplot.png')
        plt.savefig(box_filename)
        plt.close()
        print(f"Box plot saved as {box_filename}")

def main():
    """
    Main function to run the Ljung-Box test with combined data from multiple CSV files.
    """
    # Load date configuration (simplified, no error checking)
    with open('date_config.json') as file:
        config = json.load(file)
    start_date, end_date = config['START_DATE'], config['END_DATE']
    
    # Load and combine data from all input files
    all_stock_data = pd.DataFrame()
    for file in INPUT_FILES:
        try:
            print(f"Loading data from {file}...")
            data = pd.read_csv(file)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            all_stock_data = pd.concat([all_stock_data, data], ignore_index=False)
        except FileNotFoundError:
            print(f"Error: {file} not found. Skipping.")
            continue
    
    if all_stock_data.empty:
        print("Error: No valid data loaded from any input files.")
        return
    
    # Run Ljung-Box test
    print("Running Ljung-Box test...")
    results = ljung_box_test(all_stock_data, num_tickers_to_run='all')
    
    # Save results to CSV
    print("Saving test results to CSV...")
    results_data = []
    lags = [1, 3, 5, 7, 10, 15, 20, 30, 45, 60]
    for category, results_list in results.items():
        for ticker, lb_stats, p_values, randomness in results_list:
            row = [category, ticker]
            for lag in lags:
                row.append(lb_stats[f'lag_{lag}'])
                row.append(p_values[f'lag_{lag}'])
            row.append(randomness)
            results_data.append(row)
    columns = ['Category', 'Ticker'] + [f'LB_Stat_Lag_{lag}' for lag in lags] + [f'P_Value_Lag_{lag}' for lag in lags] + ['Randomness']
    results_df = pd.DataFrame(results_data, columns=columns)
    results_df.to_csv('ljung_box_results.csv', index=False)
    print("Test results saved as ljung_box_results.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_ljung_box_results(results)

if __name__ == "__main__":
    main()