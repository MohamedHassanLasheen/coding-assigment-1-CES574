import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read a CSV file (winequality-red) into a DataFrame
dataset = pd.read_csv('winequality-red.csv')

# Define the X & Y for the data set X (input) and Y (target)(pH value)
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]].values
Y = dataset.iloc[:, 8].values

# Check for missing values and decide how to handle them (e.g., impute or drop)
missing_values = dataset.isnull().sum()
# Handle missing values here
dataset.dropna(inplace=True)
# Use the describe() method to get statistics (Mean, Max, Min & Std deviation)
statistics = dataset.describe()

# Print the statistics
print(statistics)

# Display the first 5 rows of the DataFrame
print(dataset.head())

# Display information about the DataFrame
print(dataset.info())

# Normalization for the data using standardization method, calculate Mean & std_dev
mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)
standardized_data = (X - mean) / std_dev

# Print the standardized_data
print("\nStandardized Data:")
print(standardized_data)

# Divide the dataset into 80% training and 20% testing
train_ratio = 0.8
test_ratio = 0.2

# Calculate the number of samples for each split
num_samples = len(dataset)
num_train_samples = int(train_ratio * num_samples)
num_test_samples = num_samples - num_train_samples

# Randomly shuffle the data by shuffling the index array
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split the data into training and testing sets based on shuffled indices
train_indices = indices[:num_train_samples]
test_indices = indices[num_train_samples:]

train_data = dataset.iloc[train_indices, :]
test_data = dataset.iloc[test_indices, :]

# Identify the X and Y for training data
x_train = train_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]].values
y_train = train_data.iloc[:, 8].values

# Identify the X and Y for testing data
x_test = test_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]].values
y_test = test_data.iloc[:, 8].values

# Print x, y for both training and testing
print("x-train Data:")
print(x_train)
print("y-train Data:")
print(y_train)
print("x-test Data:")
print(x_test)
print("y-test Data:")
print(y_test)

# Use closed for solution for linear regression problem

# add ones column so that remove bias
X_new = np.column_stack((np.ones(x_train.shape[0]), x_train))

# get the matrix transpose
X_new_Transpose = np.transpose(X_new)

# get the matrix multiplication
XT_X = np.dot(X_new_Transpose, X_new)

# get the matrix inverse
XT_X_inv = np.linalg.inv(XT_X)

# Print the weights
w = np.dot(np.dot(XT_X_inv, X_new_Transpose), y_train)

# Calculate w (parameters using the closed-form solution
print("Weight vector is ")
print(w)

# add ones column so that remove bias to x_testing
X_test_new = np.column_stack((np.ones(x_test.shape[0]), x_test))

# Get the prediction on the testing values
y_predicted = X_test_new @ w

print("y_predicted:")
print(y_predicted)

# Get the prediction on the testing values
mse = np.mean((y_predicted - y_test) ** 2)

print("mse:")
print(mse)

# Check the shapes of the arrays
print("x_test shape:", x_test.shape)
print("y_predicted shape:", y_predicted.shape)
print("y_test shape:", y_test.shape)

# Plot the predicted values against the actual test values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_predicted, label="Data", alpha=1, color='blue')  # Set the scatter color to blue
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],label="y=x", color='red', linestyle='-')
plt.title('Predicted vs. Target Values')
plt.xlabel('Target Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
