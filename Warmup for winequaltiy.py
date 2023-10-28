import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read a CSV file into a DataFrame
dataset = pd.read_csv('winequality-red.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , -1].values

# Print the shape and type of dataset
print(dataset.shape)
print(dataset.dtypes)

# Check for missing values in the entire DataFrame
missing_values = dataset.isnull().sum()

# Print the count of missing values for each column
print(missing_values)

# Drop the missing rows
dataset.dropna(inplace=True)

# Use the describe() method to get statistics
statistics = dataset.describe()

# Print the statistics
print(statistics)

# Display the first 10 rows of the DataFrame
print(dataset.head(5))

# Display information about the DataFrame
print(dataset.info())

# Visualization 1: Histogram of 'fixed acidity'
plt.figure(figsize=(6, 6))
plt.hist(dataset['fixed acidity'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('fixed acidity')
plt.ylabel('Frequency')
plt.title('Histogram of fixed acidity')
plt.show()

# Visualization 1: Histogram of 'PH'
plt.figure(figsize=(6, 6))
plt.hist(dataset['pH'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.title('Histogram of pH')
plt.show()

# Visualization 1: Histogram of 'quality'
plt.figure(figsize=(6, 6))
plt.hist(dataset['quality'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('quality')
plt.ylabel('Frequency')
plt.title('Histogram of quality')
plt.show()

# scatter plot of pH vs quality
plt.figure(figsize=(8, 6))
sns.scatterplot(data=dataset, x='pH', y='quality')
plt.title('pH vs quality')
plt.xlabel('pH')
plt.ylabel('quality')
plt.show()

# box plot of pH
plt.figure(figsize=(8, 6))
sns.boxplot(y='pH', data=dataset, color='purple')
plt.ylabel('pH')
plt.title('Box Plot of a pH')
plt.show()

# Print running finished
print("finish running")