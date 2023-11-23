import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size):
    """Compute moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def ensure_unique_time(time):
    """Ensure time values are unique and strictly increasing."""
    for i in range(1, len(time)):
        if time[i] <= time[i-1]:
            time[i] = time[i-1] + 1e-10  # Add a very small value
    return time
# Load the Excel file
file_path = "result/pressureData.xlsx"
xls = pd.ExcelFile(file_path)

# Create a directory for the 04plots if it doesn't exist
if not os.path.exists("04plots"):
    os.makedirs("04plots")

# Iterate over each sheet in the Excel file
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)

    # Extract time and BHP columns
    time = df['time'].values
    BHP = df['BHP'].values
    # Ensure time values are unique and strictly increasing
    time = ensure_unique_time(time)

    # Smooth the BHP data using moving average
    window_size = 5  # You can adjust this value
    smoothed_BHP = moving_average(BHP, window_size)
    smoothed_time = time[window_size-1:]

    # Calculate first derivative
    p_prime = np.gradient(smoothed_BHP, smoothed_time)

    # Calculate second derivative
    p_double_prime = np.gradient(p_prime, smoothed_time)

    # Plot pressure vs. time
    plt.figure()
    plt.semilogx(smoothed_time, smoothed_BHP)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels
    plt.title(sheet_name)
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.savefig(f"04plots/{sheet_name}_pressure_vs_time.png")

    # Plot first derivative of pressure vs. time
    plt.figure()
    plt.semilogx(smoothed_time, p_prime)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels
    plt.title(sheet_name)
    plt.xlabel('Time')
    plt.ylabel("Pressure' (First Derivative)")
    plt.savefig(f"04plots/{sheet_name}_first_derivative.png")

    # Plot second derivative of pressure vs. time
    plt.figure()
    plt.semilogx(smoothed_time, p_double_prime)
    plt.title(sheet_name)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels
    plt.xlabel('Time')
    plt.ylabel("Pressure'' (Second Derivative)")
    plt.savefig(f"04plots/{sheet_name}_second_derivative.png")

    plt.close('all')  # Close all figures to free up memory
