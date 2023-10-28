import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read a CSV file into a DataFrame
dataset = pd.read_csv('diamonds.csv')
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

# Visualization 1: Histogram of 'color'
plt.figure(figsize=(6, 6))
plt.hist(dataset['color'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('color')
plt.ylabel('Frequency')
plt.title('Histogram of color')
plt.show()

# Visualization 1: Histogram of 'depth'
plt.figure(figsize=(6, 6))
plt.hist(dataset['depth'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('depth')
plt.ylabel('Frequency')
plt.title('Histogram of depth')
plt.show()

# Visualization 1: Histogram of 'table'
plt.figure(figsize=(6, 6))
plt.hist(dataset['table'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('table')
plt.ylabel('Frequency')
plt.title('Histogram of table')
plt.show()

# scatter plot of carat by depth
plt.figure(figsize=(8, 6))
sns.scatterplot(data=dataset, x='carat', y='depth')
plt.title('carat vs quality')
plt.xlabel('carat')
plt.ylabel('depth')
plt.show()

# box plot of feature
plt.figure(figsize=(8, 6))
sns.boxplot(y='depth', data=dataset, color='purple')
plt.ylabel('depth')
plt.title('Box Plot of a depth')
plt.show()

# Print running finished
print("finish running")