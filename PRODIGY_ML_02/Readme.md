# Customer Segmentation Using K-Means Clustering

## Project Overview

This repository contains the implementation of K-Means Clustering to segment customers of a retail store based on their Annual Income and Spending Score. The goal is to group customers into clusters for targeted marketing strategies.

## Index
- [Project Overview](#project-overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Steps to Run](#steps-to-run)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Results](#results)
- [Dataset](#dataset)
- [License](#license)

## Features

### Data Preprocessing
- Standardizing the data (scaling the features)
- Handling missing values (if any)

### Clustering
- Applying K-Means Clustering to divide customers into 3 distinct clusters based on Annual Income and Spending Score

### Visualization
- Scatter plot to visualize the customer segments based on the two features
- Count plot to show the distribution of customers in each cluster
- Histograms to show the distribution of distances from each customer to the centroids

### Predictions
- Assigning a cluster to each customer
- Saving the results with the distances from the centroids to a CSV file

## File Structure
```
│   task2.py
│   mall_with_distances.csv
│   mall.csv
│
└───visualizations
    └───scatter_plot.png
    └───count_plot.png
    └───distance_histogram1.png
    └───distance_histogram2.png
    └───distance_histogram3.png
```

## Steps to Run

### 1. Data Preprocessing
Run the `task2.py` script to load the data, preprocess it, and standardize the features for clustering:

```bash
python task2.py
```

This will process the customer data and apply K-means clustering. The results, including customer assignments to clusters and distances from each centroid, will be saved in the `mall_with_distances.csv` file.

### 2. K-Means Clustering and Prediction
The script will also plot and visualize the customer segments and the distances from centroids:

- A scatter plot will be generated to visualize customer segments
- A count plot will show the distribution of customers across clusters
- Histograms will display the distribution of distances to each of the centroids

All plots will be saved as PNG files in the `visualizations/` folder.

## Visualization

The project includes several visualizations that are automatically generated when running the `task2.py` script:

- **Scatter Plot**: Shows customer segmentation based on Annual Income and Spending Score
- **Count Plot**: Displays the distribution of customers in each of the 3 clusters
- **Distance Histograms**: Depict the distribution of distances of each customer from the centroids of the 3 clusters

These plots will be saved in the `visualizations/` folder and also displayed during execution.

## Dependencies

The following Python libraries are required for running this project:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Ensure these libraries are installed in your Python environment:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Results

- The K-means clustering will assign each customer to one of the three clusters based on their Annual Income and Spending Score
- Results are saved in the `mall_with_distances.csv` file, which includes:
  - Customer IDs
  - Assigned cluster label
  - Distances to each of the 3 centroids

The visualizations will help in understanding how well the customers are grouped, and the cluster assignments can be used for targeted marketing.

## Dataset

The dataset used in this project is the Mall Customer Segmentation Data, which contains information about customers' income and spending score. 

**Source**: [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

The dataset is included in the repository as `mall.csv`.

## License

This project is licensed under the MIT License.
