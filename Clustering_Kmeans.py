# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:04:52 2025

@author: rwats
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_excel('marketing_campaign.xlsx')

# Display the first few rows

df = df.dropna()

df.plot.scatter(x='Income', y='NumWebPurchases', c='blue')
plt.show()

z_scores = np.abs(stats.zscore(df[['Income', 'NumWebPurchases', 'MntWines']]))
df = df[(z_scores < 3).all(axis=1)]


# Select relevant features for clustering
features = df[['Income', 'Recency', 'MntWines', 'NumWebPurchases']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Check the scaled data
print(scaled_features[:5])

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the inertia vs number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Show the first few rows with cluster assignments
print(df[['ID', 'Income', 'Recency', 'MntWines', 'NumWebPurchases', 'Cluster']].head())

plt.figure(figsize=(8,6))
plt.scatter(df['Income'], df['NumWebPurchases'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering (Income vs NumWebPurchases)')
plt.xlabel('Income')
plt.ylabel('NumWebPurchases')
plt.show()

# Cluster Centroids: The centroids of the clusters show the average values 
# for each feature (e.g., income, recency, purchase behavior). 
# For example, one cluster might have higher income, 
# while another may have more frequent purchases but lower income.

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
print(centroids)

# Summarize the data for each cluster
cluster_summary = df.groupby('Cluster').agg({
    'Income': ['mean', 'std'],
    'Recency': ['mean', 'std'],
    'MntWines': ['mean', 'std'],
    'NumWebPurchases': ['mean', 'std']
})
print(cluster_summary)



# Cluster 0 (Centroid 0):

# Income: -0.699384 — The income in this cluster is below average (as indicated by the negative value when scaled).
# Recency: 0.027363 — The recency value is near the average, indicating that customers in this cluster have moderate recent engagement with the business.
# MntWines: -0.774559 — Customers in this cluster spend less on wine compared to other clusters (negative value when scaled).
# NumWebPurchases: -0.726522 — These customers also make fewer web purchases, again indicated by the negative scaling value.

# Cluster 1 (Centroid 1):

# Income: 0.671021 — Customers in this cluster have higher-than-average income (positive scaling value).
# Recency: 0.825125 — These customers have been more recently engaged with the business, as indicated by the higher positive value.
# MntWines: 0.806325 — Customers in this cluster spend significantly more on wine, reflecting a high positive value.
# NumWebPurchases: 0.644187 — Customers also make more web purchases than others, as indicated by the positive value.

# Cluster 2 (Centroid 2):

# Income: 0.582440 — Customers here have higher-than-average income, though not as high as Cluster 1.
# Recency: -0.951981 — These customers have lower recency, meaning they have not interacted with the business recently (negative scaling value).
# MntWines: 0.576093 — These customers spend moderately on wine.
# NumWebPurchases: 0.662746 — These customers are somewhat active in terms of web purchases but not as much as those in Cluster 1.
