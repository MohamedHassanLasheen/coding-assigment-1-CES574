import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read a CSV file into a DataFrame
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 6].values

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
print(dataset.head(10))

# Display information about the DataFrame
print(dataset.info())

# Convert 'Category' column to categorical
dataset['sex'] = dataset['sex'].astype('category')
dataset['smoker'] = dataset['smoker'].astype('category')
dataset['region'] = dataset['region'].astype('category')

# Check the data types
print(dataset.dtypes)

# Visualization 1: Histogram of 'sex'
plt.figure(figsize=(6, 6))
plt.hist(dataset['sex'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('sex')
plt.ylabel('Frequency')
plt.title('Histogram of sex')
plt.show()

# Visualization 1: Histogram of 'smoker'
plt.figure(figsize=(6, 6))
plt.hist(dataset['smoker'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('smoker')
plt.ylabel('Frequency')
plt.title('Histogram of smoker')
plt.show()

# Visualization 1: Histogram of 'children'
plt.figure(figsize=(6, 6))
plt.hist(dataset['children'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('children')
plt.ylabel('Frequency')
plt.title('Histogram of children')
plt.show()

# scatter plot of bmi by age
plt.figure(figsize=(8, 6))
sns.scatterplot(data=dataset, x='sex', y='age')
plt.title('sex vs age')
plt.xlabel('sex')
plt.ylabel('age')
plt.show()

# box plot of feature
plt.figure(figsize=(8, 6))
sns.boxplot(y='charges', data=dataset, color='purple')
plt.ylabel('charges')
plt.title('Box Plot of a charges')
plt.show()

# Print running finished
print("finish running")