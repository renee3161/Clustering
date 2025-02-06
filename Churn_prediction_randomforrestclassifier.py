# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:09:40 2025

@author: rwats
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_excel('marketing_campaign.xlsx')  # replace with actual data path

# Check for missing values
print(df.isnull().sum())

# Fill missing data if necessary (example)
df['Income'] = df['Income'].fillna(df['Income'].mean())  # Example: fill missing 'Income' with mean value
df['MntWines'] = df['MntWines'].fillna(df['MntWines'].mean())

# Drop any irrelevant columns, like 'ID' or 'Dt_Customer' (date-related features)
df.drop(columns=['ID', 'Dt_Customer'], inplace=True)

# Handle categorical columns (Label Encoding for simplicity)
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

X = df[['Income', 'Recency', 'MntWines', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
y = df['Response']  # Assuming 'Response' is the target variable for churn (1 for churn, 0 for non-churn)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)

# Evaluate the model
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example new customers
new_customers = pd.DataFrame({
    'Income': [55000, 72000, 43000],
    'Recency': [40, 10, 60],
    'MntWines': [200, 300, 150],
    'NumWebPurchases': [5, 8, 2],
    'NumStorePurchases': [4, 1, 6],
    'NumWebVisitsMonth': [10, 12, 5]
})

# Scale the new data
new_customers_scaled = scaler.transform(new_customers)

# Predict churn (1 for churn, 0 for non-churn)
predictions = rf.predict(new_customers_scaled)
print(predictions)  # Output will be a list of 1's and 0's representing churn status
