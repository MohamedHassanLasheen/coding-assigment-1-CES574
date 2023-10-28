import numpy as np
import pandas as pd
from logistic import LogitRegression

# Read a CSV file (penguins) into a DataFrame
dataset = pd.read_csv('penguins.csv')

# Determine how many row has NaN or missing value
missing_values = dataset.isnull().sum()

#  Drop the NaN rows or any row has missing value
dataset.dropna(inplace=True)

# Use the describe() method to get statistics (Mean, Max, Min & Std deviation)
print(dataset.describe())

# Display the first 5 rows of the DataFrame
print(dataset.head(5))

# Display information about the DataFrame
print(dataset.info())

# Define a list of column names that you want to convert to categorical
categorical_columns = ['species', 'island', 'sex']

# Convert the specified columns to categorical
for column in categorical_columns:
    dataset[column] = pd.Categorical(dataset[column])

# Convert categorical columns to binary values
for column in categorical_columns:
    dataset[column] = dataset[column].cat.codes

# Determine the name for non_categorical_columns
non_categorical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Normalize the data by rescaling each column
min_values = dataset[non_categorical_columns].min()
max_values = dataset[non_categorical_columns].max()
normalized_data = (dataset[non_categorical_columns] - min_values) / (max_values - min_values)

# Recombine the normalized and categorical data after being binary into new array
dataset_new = np.hstack((normalized_data, dataset[categorical_columns]))

# Define the X & Y for the data set X (input) and Y (target)
X = dataset_new[:, :-1]
Y = dataset_new[:, -1]

# Divide the dataset into 80% training and 20% testing
train_ratio = 0.8
test_ratio = 0.2

# Calculate the number of samples for each split
num_samples = len(dataset_new)
num_train_samples = int(train_ratio * num_samples)
num_test_samples = num_samples - num_train_samples

# give index for each sample
indices = np.arange(num_samples)

# Split the data into training and testing sets based on shuffled indices
train_indices = indices[:num_train_samples]
test_indices = indices[num_train_samples:]

train_data = dataset_new[train_indices, :]
test_data = dataset_new[test_indices, :]

# Identify the X and Y for training data
x_train = train_data[:, :-1]
y_train = train_data[:, -1]

# Identify the X and Y for testing data
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Print shape of x_train,y_train , x_test and y_test
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Initialize and train the logistic regression model for case II for example (i change this 3 times)
learning_rate = 1e-3
iterations = 10000
model = LogitRegression(learning_rate, iterations)

# Train the logistic regression model
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)
print("Predictions:", predictions)

# Count correct and incorrect predictions
correct_predictions = (predictions == y_test).sum()
incorrect_predictions = len(y_test) - correct_predictions

# Calculate accuracy
accuracy = correct_predictions / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

