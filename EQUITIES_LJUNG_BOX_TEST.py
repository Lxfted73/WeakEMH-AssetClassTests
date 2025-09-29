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
    Create a stacked bar plot using Seaborn/Matplotlib showing the proportion of tickers with
    significant (p < 0.05) vs. non-significant (p >= 0.05) Ljung-Box p-values for selected lags
    across categories, with distinct colors for each category. Save plot data to a CSV file
    including the number of companies per category.
    
    Parameters:
    - results_by_category: Dictionary with categories as keys and lists of [ticker, lb_stats, p_values, randomness]
    - output_dir: Directory to save the plot image and data CSV
    """
    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Select subset of lags to reduce complexity
    selected_lags = [1, 10, 60]
    
    # Define colors for each category (updated with darker significant and brighter non-significant shades)
    colors = {
        'Emerging': {'significant': '#1b5e20', 'non_significant': '#66bb6a'},
        'Large-cap': {'significant': '#0d47a1', 'non_significant': '#4fc3f7'},
        'Mid-cap': {'significant': '#4a148c', 'non_significant': '#ce93d8'},
        'Random': {'significant': '#d84315', 'non_significant': '#ffb300'},
        'Small-cap': {'significant': '#b71c1c', 'non_significant': '#ef9a9a'}
    }
    
    # Prepare data for stacked bar plot
    plot_data = []
    for category in sorted(results_by_category.keys()):
        if category not in colors:
            print(f"Warning: No color defined for category {category}. Skipping.")
            continue
        # Get total number of companies (tickers) for the category
        total = len(results_by_category[category])
        print(f"Category {category}: {total} companies")
        for lag in selected_lags:
            # Count significant (p < 0.05) and non-significant (p >= 0.05) p-values
            significant = sum(1 for _, _, p_values, _ in results_by_category[category] if p_values[f'lag_{lag}'] < 0.05)
            non_significant = total - significant
            # Calculate proportions
            if total > 0:
                plot_data.append({
                    'Category': category,
                    'Lag': f'Lag {lag}',
                    'Significant': significant / total,
                    'Non-significant': non_significant / total,
                    'Num_Companies': total
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Save plot data to CSV
    csv_filename = os.path.join(output_dir, 'ljung_box_plot_data.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Plot data saved as {csv_filename}")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot stacked bars
    categories = sorted(df['Category'].unique())
    lags = df['Lag'].unique()
    bar_width = 0.15  # Adjust width based on number of categories
    x_positions = np.arange(len(lags))
    
    for i, category in enumerate(categories):
        cat_data = df[df['Category'] == category]
        # Plot significant bars
        plt.bar(
            x_positions + i * bar_width,
            cat_data['Significant'],
            bar_width,
            label=f'{category}: Significant (p < 0.05)',
            color=colors[category]['significant'],
            edgecolor='white'
        )
        # Plot non-significant bars on top
        plt.bar(
            x_positions + i * bar_width,
            cat_data['Non-significant'],
            bar_width,
            bottom=cat_data['Significant'],
            label=f'{category}: Non-significant (p >= 0.05)',
            color=colors[category]['non_significant'],
            edgecolor='white'
        )
    
    # Customize plot
    plt.xticks(x_positions + bar_width * (len(categories) - 1) / 2, lags)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Proportion of Tickers', fontsize=12)
    plt.title('Proportion of Tickers with Significant vs. Non-significant P-values by Category', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    bar_filename = os.path.join(output_dir, 'ljung_box_stacked_bar.png')
    plt.savefig(bar_filename, bbox_inches='tight')
    plt.close()
    print(f"Stacked bar plot saved as {bar_filename}")

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