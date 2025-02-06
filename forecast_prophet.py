# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:36:44 2025

@author: Inventory
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

# Step 1: Read the data from an Excel file
df = pd.read_excel("product_sold_quantities.xlsx", header=0)

# Check the first few rows to understand the data structure
print(df.head())

# Step 2: Prepare the data
# Assume the first column is the Product ID and subsequent columns are monthly sales data
# Ensure that the DataFrame has columns starting from the second column (index 1) for the monthly data
if df.shape[1] > 1:  # Ensuring there are multiple columns beyond just Product IDs
    products = df.iloc[:, 0]  # Product IDs
    dates = df.columns[1:]    # Dates (columns except the first one)
    sales_data = df.iloc[:, 1:].replace(0, np.nan)  # Replace zero values with NaN (missing data)
else:
    raise ValueError("The dataset seems to have only one column or no monthly data!")

# Step 3: Forecast using Exponential Smoothing for each product
forecasted_data = []

for product in range(sales_data.shape[0]):
    # Extract the sales data for the product
    product_sales = sales_data.iloc[product].dropna()  # Drop NaN values (missing months)
    
    # Apply Exponential Smoothing only if there is data available for the product
    if len(product_sales) > 1:  # We need more than 1 data point to apply smoothing
        model = ExponentialSmoothing(product_sales, trend='add', seasonal='add', damped_trend=False)
        model_fit = model.fit()
        
        # Forecast for the next 12 months (one year)
        forecast = model_fit.forecast(12)
        
        # Combine the existing sales data and forecast
        complete_forecast = np.concatenate([product_sales.values, forecast])
    else:
        # If not enough data, just copy the available data and append zeros for the forecast
        complete_forecast = np.concatenate([product_sales.values, np.zeros(12)])
    
    # Append the forecasted values for this product
    forecasted_data.append(complete_forecast)

# Step 4: Create a new DataFrame with the forecasted data
forecast_df = pd.DataFrame(forecasted_data, columns=list(dates) + [f"Forecast_{i+1}" for i in range(12)])
forecast_df.insert(0, "Product", products)

# Step 5: Write the forecasted quantities to a new Excel file
forecast_df.to_excel("sales_forecast.xlsx", index=False, engine='openpyxl')

print("Sales forecast has been generated and saved to 'sales_forecast.xlsx'")




import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

# Step 1: Read the data from an Excel file
df = pd.read_excel("product_sold_quantities.xlsx", header=0)

# Step 2: Prepare the data
# Assume the first column is the Product ID and subsequent columns are monthly sales data
# Ensure that the DataFrame has columns starting from the second column (index 1) for the monthly data
if df.shape[1] > 1:  # Ensuring there are multiple columns beyond just Product IDs
    products = df.iloc[:, 0]  # Product IDs
    dates = df.columns[1:]    # Dates (columns except the first one)
    sales_data = df.iloc[:, 1:].replace(0, np.nan)  # Replace zero values with NaN (missing data)
else:
    raise ValueError("The dataset seems to have only one column or no monthly data!")

# Step 3: Forecast using Exponential Smoothing for each product
forecasted_data = []

for product in range(sales_data.shape[0]):
    # Extract the sales data for the product
    product_sales = sales_data.iloc[product]
    
    # Apply Exponential Smoothing only if there is data available for the product
    valid_data = product_sales.dropna()  # Drop NaN values (missing months)

    if len(valid_data) > 1:  # We need more than 1 data point to apply smoothing
        model = ExponentialSmoothing(valid_data, trend='add', seasonal=None, damped_trend=False)
        model_fit = model.fit()

        # Forecast for the next 12 months (one year)
        forecast = model_fit.forecast(12)
        
        # Combine the original sales data with the forecasted values for the next 12 months
        forecasted_values = np.concatenate([product_sales.values, forecast])
    else:
        # If not enough data, just copy the available data and append zeros for the forecast
        forecasted_values = np.concatenate([product_sales.values, np.zeros(12)])
    
    # Append the forecasted values for this product to the final forecasted data
    forecasted_data.append(forecasted_values)

# Step 4: Create a new DataFrame with the forecasted data
forecast_df = pd.DataFrame(forecasted_data, columns=list(dates) + [f"Forecast_{i+1}" for i in range(12)])
forecast_df.insert(0, "Product", products)

# Step 5: Write the forecasted quantities to a new Excel file
forecast_df.to_excel("sales_forecast_corrected.xlsx", index=False, engine='openpyxl')

print("Sales forecast has been generated and saved to 'sales_forecast_corrected.xlsx'")


