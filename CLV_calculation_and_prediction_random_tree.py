# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:01:45 2025

@author: rwats
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Read excel file
df = pd.read_excel('marketing_campaign.xlsx')

# Calculate Average Order Value (AOV)
# We'll calculate AOV as the average spending across different product categories (e.g., Wine, Meat)
df['AOV'] = df[['MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 'MntSweetProducts']].mean(axis=1)

# Calculate Purchase Frequency
# Assuming 'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases' are indicators of purchases
df['purchase_frequency'] = df[['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']].sum(axis=1)

# Calculate Customer Lifespan (based on Recency)
# Customer lifespan can be approximated by subtracting 'Recency' from a defined maximum customer lifetime (e.g., 1000 days)
df['customer_lifespan'] = 1000 - df['Recency']  # You can adjust this depending on your business

# Calculate CLV
# Apply the simple CLV formula
df['CLV'] = df['AOV'] * df['purchase_frequency'] * df['customer_lifespan']
df.to_excel('Customer_lifetime_Value.xlsx') # save CLV to excel

# Display the calculated CLV for each customer
print(df[['ID', 'CLV']])

# Identify high-value customers:
high_value_customers = df[df['CLV'] > df['CLV'].quantile(0.75)]  # Top 25% of customers
high_value_customers.to_excel('High_value_customer.xlsx')
print(high_value_customers[['ID', 'CLV']])


# Features and target (CLV in this case)
X = df[['Income', 'Recency', 'NumWebPurchases', 'NumStorePurchases', 'MntWines', 'MntMeatProducts']]
y = df['CLV']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict CLV on the test set
y_pred = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
print (y_pred)

# Evaluate the model
print(f"R-squared: {model.score(X_test, y_test)}")

new_customers = pd.DataFrame({
    'Income': [55000, 72000, 43000],
    'Recency': [40, 10, 60],
    'NumWebPurchases': [5, 8, 2],
    'NumStorePurchases':[4, 1, 6],
    'MntWines': [200, 300, 150],
    'MntMeatProducts': [26, 258, 211]
})


clv_predictions = model.predict(new_customers)

# Display the predictions
print("Predicted CLV for new customers:", clv_predictions)