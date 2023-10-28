import numpy as np
import matplotlib.pyplot as plt

loss = []

# first define the class of name logit regression
class LogitRegression:
    # Define the initialization for learning rate and num_iteration and weight
    def __init__(self, learning_rate=1e-4, num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # the weight vector includes feature weights and bias {b , w1 , w2, ...,w7}
        self.weights = None

    # Define fit function to get the required weight vector
    def fit(self, X, y):
        # initialize the weight vector randomly using uniform dist. (0 ,1),the size (n_features + 1) to include bias
        self.weights = np.random.uniform(0, 1, X.shape[1] + 1)

        # iterate to adjust the weights using gradient descent and print the loss every 1000 iteration
        for i in range(self.num_iterations):
            self.gradient_descent(X, y)
            loss.append(self.cost(X, y))
            if i % 1000 == 0:
                print(self.cost(X, y))

        # Check if it's the last iteration and print the final weight vector
        if i == self.num_iterations - 1:
            print(self.weights)

        # plot the loss vs iteration
        plt.figure(1, figsize=(6, 5))
        plt.plot(loss)
        plt.grid()
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.show()

    # Define sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Define loss function for binary logistic classification sum for all samples
    def cost(self, X, y):
        # Define new x array with ones vector (X,1) and combine with X matrix in combined matrix
        combined_X = np.hstack((X, np.ones((X.shape[0], 1))))
        # Calculate z vector = combined_x * Weight vector, z vector is vector containing z for each sample
        z = np.matmul(combined_X, self.weights)
        # Calculate y_pred vector, y_pred vector containing y_pred for each sample
        y_pred = self.sigmoid(z)
        # Calculate average loss and add small value to log to avoid NaN if y_pred = 1
        epsilon = 1e-15
        m, n_features = X.shape
        loss = - (1 / m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss

    # Define loss function for binary logistic classification sum for all samples
    def gradient_descent(self, X, y):
        # Define new x array with ones vector (X,1)
        combined_X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Calculate z vector = combined_X * Weight vector
        z = np.matmul(combined_X, self.weights)

        # Calculate y_pred vector, y_pred vector containing y_pred for each sample
        y_pred = self.sigmoid(z)

        # Calculate gradient vector
        dw = np.matmul(combined_X.T, (y_pred - y))

        # calculate adjusted weights for each iteration
        self.weights -= self.learning_rate * dw

    # Define prediction function to make prediction
    def predict(self, X):
        combined_X = np.hstack((X, np.ones((X.shape[0], 1))))
        z = np.matmul(combined_X, self.weights)

        # Calculate y_pred vector
        y_pred = self.sigmoid(z)

        # make the final prediction
        binary_predictions = np.where(y_pred >= 0.5, 1, 0)
        return binary_predictions
