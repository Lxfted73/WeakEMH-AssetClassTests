import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# List of input CSV files (add or remove files to include additional categories)
INPUT_FILES = ['data_all_equities.csv', 'data_random_walk.csv']

def plot_zscore_boxplot(results_by_category, output_dir='runs_test_plots'):
    """
    Generate a box plot of Z-scores across all categories with improved colors.
    
    Parameters:
    - results_by_category: Dictionary with categories as keys and lists of [ticker, runs, expected_runs, z_score, randomness] as values
    - output_dir: Directory to save the plot image
    """
    # Prepare data for box plot
    all_z_scores = {cat: [res[3] for res in results_by_category[cat]] for cat in results_by_category if results_by_category[cat]}
    
    if not all_z_scores:
        print("No data available for box plot.")
        return
    
    # Set seaborn style and color palette
    sns.set_style("whitegrid")
    palette = sns.color_palette("muted", n_colors=len(all_z_scores))
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(all_z_scores.values(), labels=all_z_scores.keys(), patch_artist=True)
    
    # Apply colors to boxes
    for patch, color in zip(box['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # Customize whiskers, caps, and medians
    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in box['caps']:
        cap.set(color='black', linewidth=1.5)
    for median in box['medians']:
        median.set(color='black', linewidth=2)
    
    # Add significance lines
    plt.axhline(y=1.96, color='#003087', linestyle='--', label='±1.96 (95% significance)')
    plt.axhline(y=-1.96, color='#003087', linestyle='--')
    
    plt.xlabel('Category')
    plt.ylabel('Z-score')
    plt.title('Z-score Distribution Across Categories')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    box_filename = os.path.join(output_dir, 'test_results_runs.png')
    plt.savefig(box_filename)
    plt.close()
    print(f"Box plot saved as {box_filename}")

def runs_test(all_stock_data, num_tickers_to_run='all'):
    """
    Perform a runs test on stock price data to assess randomness of returns, organized by category.
    
    Parameters:
    - all_stock_data: DataFrame with columns ['Category', 'Ticker', 'Date', 'Close']
    - num_tickers_to_run: 'all' or an integer to limit the number of tickers processed per category
    
    Returns:
    - Dictionary with categories as keys and lists of [ticker, runs, expected_runs, z_score, randomness] as values
    """
    # Validate input data
    required_columns = ['Category', 'Ticker', 'Date', 'Close']
    if not all(col in all_stock_data.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

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
                
                # Calculate returns and signs
                prices = stock_data['Close']
                returns = prices.pct_change().dropna()  # Calculate percentage change
                signs = ['+' if r > 0 else '-' for r in returns]
                
                # Perform runs test
                n1 = signs.count('+')  # Number of positive returns
                n2 = len(signs) - n1   # Number of negative returns
                n = n1 + n2
                expected_runs = (2 * n1 * n2) / n + 1 if n > 0 else 0
                runs = 1 + sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
                variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (n**2 * (n-1)) if n > 1 else 0
                z = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
                
                # Store results
                result_entry = [ticker, runs, f"{expected_runs:.2f}", f"{z:.2f}",
                               'Non-random' if abs(z) > 1.96 else 'Random']
                category_results.append(result_entry)
                
                # Add to results dictionary
                if category not in results_by_category:
                    results_by_category[category] = []
                results_by_category[category].append([ticker, runs, expected_runs, z,
                                                    'Non-random' if abs(z) > 1.96 else 'Random'])
                
            except Exception as e:
                print(f"{ticker}: Error processing data - {e}")
                continue
        
        # Pretty print table for category
        if category_results:
            print(tabulate(category_results,
                          headers=['Ticker', 'Runs', 'Expected Runs', 'Z-score', 'Randomness'],
                          tablefmt='fancy_grid'))
        else:
            print(f"No valid data for tickers in category {category}")
            
    plot_zscore_boxplot(results_by_category, "")
    return results_by_category 

def main():
    """
    Main function to run the runs test with combined data from multiple CSV files.
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
    
    # Run the test
    print("Running runs test...")
    results = runs_test(all_stock_data, num_tickers_to_run='all')

if __name__ == "__main__":
    main()