import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')

features = df[['AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['ClusterAssigned'] = kmeans.fit_predict(scaled_features)

# Create a new DataFrame to store the distances to each centroid
distances = pd.DataFrame(columns=['DistanceToCentroid1', 'DistanceToCentroid2', 'DistanceToCentroid3'])

for i in range(3):
    centroid = kmeans.cluster_centers_[i]
    dist_to_centroid = np.linalg.norm(scaled_features - centroid, axis=1)  # Euclidean distance
    distances[f'DistanceToCentroid{i+1}'] = dist_to_centroid

df = pd.concat([df, distances], axis=1)

# Save the updated DataFrame to csv file mall_with_distances.csv
df.to_csv('mall_with_distances.csv', index=False)

# Visualization

# 1. Scatter plot of clusters
print("\nPlot Name: Scatter plot of Customer Segments (Clustered)")
print("Plot Parameters: \n- x: AnnualIncome \n- y: SpendingScore \n- hue: ClusterAssigned \n- palette: viridis \n- s: 100 (Marker size)")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='AnnualIncome', 
    y='SpendingScore', 
    hue='ClusterAssigned', 
    data=df, 
    palette='viridis', 
    legend='full', 
    s=100
)
plt.title('Customer Segments (Clustered)', fontsize=15)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)

plt.show()
print("Description: This scatter plot shows the segmentation of customers based on Annual Income and Spending Score. The colors represent different clusters identified by K-means.\n")

# 2. Distribution of customers per cluster
print("\nPlot Name: Distribution of Customers per Cluster (Bar plot)")
print("Plot Parameters: \n- x: ClusterAssigned \n- hue: ClusterAssigned \n- legend: False")
plt.figure(figsize=(8, 6))
sns.countplot(x='ClusterAssigned', data=df, hue='ClusterAssigned', legend=False)
plt.title('Number of Customers in Each Cluster', fontsize=15)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)

plt.show()
print("Description: This bar plot shows the distribution of customers across the three clusters. It indicates how many customers fall into each of the identified clusters.\n")

# 3. Distance to Centroid 1
print("\nPlot Name: Distance to Centroid 1 (Histogram)")
print("Plot Parameters: \n- Data: DistanceToCentroid1 \n- KDE: True \n- Color: Blue")
plt.figure(figsize=(8, 6))
sns.histplot(df['DistanceToCentroid1'], kde=True, color='blue')
plt.title('Distance to Centroid 1', fontsize=15)
plt.xlabel('Distance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.show()
print("Description: This histogram depicts the distribution of distances from each customer to Centroid 1. A higher peak indicates that most customers are closer to this centroid.\n")

# 4. Distance to Centroid 2
print("\nPlot Name: Distance to Centroid 2 (Histogram)")
print("Plot Parameters: \n- Data: DistanceToCentroid2 \n- KDE: True \n- Color: Orange")
plt.figure(figsize=(8, 6))
sns.histplot(df['DistanceToCentroid2'], kde=True, color='orange')
plt.title('Distance to Centroid 2', fontsize=15)
plt.xlabel('Distance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.show()
print("Description: This histogram shows the distribution of distances from each customer to Centroid 2. It helps to visualize how well the customers are grouped around this centroid.\n")

# 5. Distance to Centroid 3
print("\nPlot Name: Distance to Centroid 3 (Histogram)")
print("Plot Parameters: \n- Data: DistanceToCentroid3 \n- KDE: True \n- Color: Green")
plt.figure(figsize=(8, 6))
sns.histplot(df['DistanceToCentroid3'], kde=True, color='green')
plt.title('Distance to Centroid 3', fontsize=15)
plt.xlabel('Distance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.show()
print("Description: This histogram shows the distribution of distances from each customer to Centroid 3. This helps to identify how well the customers are clustered around this particular centroid.\n")