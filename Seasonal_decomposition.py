# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:44:40 2025

@author: rwats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Read the data into a DataFrame
df = pd.read_excel('customer_sales_formatted.xlsx', index_col='Customer')
df = df.dropna()
# Transpose the data so that months are rows and customers are columns
df = df.transpose()

# Ensure that the data is sorted by month
df = df.sort_index()

# Intialize dictionary to store the seasonal peaks for each customer
seasonal_peaks = {}

# Iterate over each customer (column)
for customer in df.columns:
    # Get the sales data for the customer
    sales_data = df[customer]

    # Perform seasonal decomposition of the sales data
    decomposition = seasonal_decompose(sales_data, model='additive', period=6)  # Assuming monthly data

    # Extract the seasonal component
    seasonal_component = decomposition.seasonal

    # Identify the peak month for seasonality (i.e., the month with the highest seasonal value)
    peak_month = seasonal_component.idxmax()

    # Save the result
    seasonal_peaks[customer] = peak_month

    # Plot the decomposition
    plt.figure(figsize=(10, 6))
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition for {customer}')
    plt.show()

# Convert the seasonal peaks to a DataFrame
seasonal_peaks_df = pd.DataFrame(list(seasonal_peaks.items()), columns=['Customer', 'Peak Seasonality Month'])

# Save the seasonal peaks to an Excel file
seasonal_peaks_df.to_excel('seasonal_peaks_2024.xlsx', index=False)

# Print out the seasonal peaks
print(seasonal_peaks_df)