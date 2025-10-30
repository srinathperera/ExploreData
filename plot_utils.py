import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_multiple_y_axes(df, x_axis_name, y_axes_names):
    """
    Plot multiple y axes.
    """
    title = x_axis_name + "_vs_" 
    for i, y_axis_name in enumerate(y_axes_names):
        title += f", {y_axis_name}_"
    title = title + ".png"
    plt.figure(figsize=(12, 6))
    plt.xlabel(x_axis_name)
    for i, y_axis_name in enumerate(y_axes_names):
        plt.plot(df[x_axis_name], df[y_axis_name], label=y_axis_name, marker='o')
    plt.legend()
    plt.title(title)
    plt.savefig("temp/" + title, dpi=300)
    print("Saved plot to temp/" + title)
    plt.close()

def scatter_plot_with_many_y_axes(df, x_axis_name, y_axes_names):
    """
    Scatter plot with many y axes.
    """
    title = x_axis_name + "_vs_" 
    for i, y_axis_name in enumerate(y_axes_names):
        title += f", {y_axis_name}_"
    title = title + ".png"

    plt.figure(figsize=(12, 6))
    plt.xlabel(x_axis_name)
    for i, y_axis_name in enumerate(y_axes_names):
        plt.scatter(df[x_axis_name], df[y_axis_name], label=y_axis_name, marker='o')
    plt.legend()
    plt.title(title)
    plt.savefig("temp/" + title, dpi=300)
    print("Saved plot to temp/" + title)
    plt.close()

def plot_aggregated_data_with_error_bars(df: pd.DataFrame, time_col: str = 't', value_cols: list = ['a', 'b'], filename: str = 'aggregated_plot.png'):
    """
    Aggregates a DataFrame by a time column and plots the means of specified 
    value columns against the time column, including standard deviation 
    as error bars on the same plot.

    Args:
        df (pd.DataFrame): The input DataFrame with columns t, a, and b.
        time_col (str): The name of the column to use as the grouping key (x-axis). Default is 't'.
        value_cols (list): A list of column names to aggregate and plot (y-axis). Default is ['a', 'b'].
        filename (str): The name of the file to save the plot to.
    """
    
    # --- 1. Aggregation ---
    # Group by the time column and calculate mean and standard deviation (std) for the value columns
    # We use std for the error bars as it represents the variability of the data for each 't' value.
    aggregated_data = df.groupby(time_col)[value_cols].agg(['mean', 'std'])
    
    # Reset index to make 't' a standard column again
    aggregated_data = aggregated_data.reset_index()
    
    # --- 2. Plotting ---
    
    # Prepare the x-axis data
    x_data = aggregated_data[time_col]
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e'] # Custom colors for 'a' and 'b'
    
    for i, col in enumerate(value_cols):
        # Extract mean and standard deviation for the current column
        y_mean = aggregated_data[(col, 'mean')]
        y_error = aggregated_data[(col, 'std')]
        
        # Plot the data with error bars
        # The 'fmt'='-o' creates a line connecting the data points with circle markers.
        ax.errorbar(
            x_data, 
            y_mean, 
            yerr=y_error, 
            fmt='-o', 
            capsize=5, # size of the caps on the error bars
            label=f'Mean of {col} (Error: $\\pm 1$ STD)',
            color=colors[i]
        )

    # --- 3. Formatting and saving ---
    ax.set_xlabel(f'{time_col.capitalize()} Value', fontsize=12)
    ax.set_ylabel('Aggregated Value (Mean)', fontsize=12)
    ax.set_title(f'Mean of {", ".join(value_cols)} vs. {time_col} with $\\pm$ Standard Deviation', fontsize=14)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    filename = "temp/" + filename
    plt.savefig(filename)
    plt.close(fig) # Close the figure to free memory
    
    print(f"Plot successfully generated and saved to {filename}")
    print("\n--- Aggregated Data Preview ---")
    print(aggregated_data.head())
    
    return aggregated_data


#show confidance instervals 
#ns.barplot()	Comparing means across categorical groups. CIs are drawn as vertical lines by default.	ci (default is 95), estimator (default is mean).
#sns.pointplot()	Similar to a bar plot, but shows the mean point and CI using caps. Better for comparison.	Automatically calculates and plots the CI.
#sns.regplot() / sns.lmplot()	Drawing a regression line and a shaded confidence band around it.	Plots the mean regression line and the confidence interval for the regression estimate.

def show_confidence_intervals_with_many_y_axes(df, x_axis_name, y_axis_names):
    """
    Show confidence intervals.
    """
    title = x_axis_name + "_vs_" 
    for i, y_axis_name in enumerate(y_axis_names):
        title += f", {y_axis_name}_"
    title = title + ".png"
    plt.figure(figsize=(12, 6))
    plt.xlabel(x_axis_name)
    for i, y_axis_name in enumerate(y_axis_names):
        sns.regplot(x=x_axis_name, y=y_axis_name, data=df, ci=95)
        #sns.pointplot(x=x_axis_name, y=y_axis_name, data=df, errorbar=('ci', 95))
    plt.legend()
    plt.title(title)
    plt.savefig("temp/ci-" + title, dpi=300)
    print("Saved plot to temp/" + title)
    plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_weighted_density_plot(df: pd.DataFrame, x_col: str, y_col: str, weight_col: str):
    """
    Generates a 2D density plot (heatmap) where the color intensity (density) 
    is weighted by a third numerical field (e.g., 'advantage').

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column for the x-axis (e.g., 'timestamp').
        y_col (str): The column for the y-axis (e.g., 'difficulty').
        weight_col (str): The column whose values determine the density/intensity (e.g., 'advantage').
        filename (str): The name of the file to save the plot to.
    """
    filename = "temp/weighted-density-" + x_col + "_vs_" + y_col + "_vs_" + weight_col + ".png"
    # 1. Input Validation (Ensure columns exist)
    required_cols = [x_col, y_col, weight_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # 2. Setup the plot
    plt.figure(figsize=(10, 7))

    # 3. Create the 2D Histogram (Heatmap) with Weights
    # plt.hist2d is used because the 'weights' parameter allows the color intensity 
    # to be determined by the sum of 'advantage' values in each bin, 
    # rather than just the raw count of points.
    H, xedges, yedges, im = plt.hist2d(
        df[x_col],
        df[y_col],
        bins=10,  # Number of bins/resolution for the grid
        weights=df[weight_col],
        cmap='viridis' # Color map for intensity
    )

    # 4. Formatting and saving
    cb = plt.colorbar(im, label=f'Total {weight_col.capitalize()} Value in Bin')
    plt.title(f'Weighted 2D Density: Total {weight_col.capitalize()} vs. {x_col.capitalize()} and {y_col.capitalize()}')
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.capitalize())
    
    plt.savefig(filename)
    plt.close()
    
