# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:36:16 2025

@author: Inventory
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_excel('marketing_campaign.xlsx')  # replace with actual data path

# Create a new target variable indicating if the customer has accepted any campaign
df['Target_accepted_campaign'] = (
    df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1) > 0
).astype(int)

# View the updated DataFrame
print(df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Target_accepted_campaign']].head())

# Select relevant features and target
X = df[['Income', 'Recency', 'MntWines', 'NumWebPurchases', 'NumStorePurchases', 'MntMeatProducts']]  # Features
y = df['Target_accepted_campaign']  # Target variable


# Fill missing values, if any (optional)
X.fillna(X.mean(), inplace=True)  # For numerical columns

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for some models)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluate the model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
y_pred = rf.predict(X_test_scaled)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example data for new customers
new_customers = pd.DataFrame({
    'Income': [89000, 72000, 43000],
    'Recency': [20, 10, 60],
    'MntWines': [400, 700, 150],
    'NumWebPurchases': [58, 58, 29],
    'NumStorePurchases': [50, 20, 30],
    'MntMeatProducts': [438, 258, 211]
})

# Scale the new customer data
new_customers_scaled = scaler.transform(new_customers)

# Predict whether they will accept a future campaign (1 = likely, 0 = unlikely)
new_predictions = rf.predict(new_customers_scaled)
print("Predictions for new customers (1 = likely, 0 = unlikely):", new_predictions)

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters: ", grid_search.best_params_)

