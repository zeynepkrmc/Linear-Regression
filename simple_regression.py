## Simple Regression
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std computed over training data.

import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling

# Read data matrix X and labels y from text file.
def read_data(file_name):
#  YOUR CODE HERE
  X = []
  y = []
  with open(file_name, 'r') as file:
      for line in file:
          # split data
          data = line.strip().split()
          if len(data) == 2:
              X.append(float(data[0]))#first col
              y.append(float(data[1]))#second col
  return np.array(X).reshape(-1, 1), np.array(y)#np.array(X), np.array(y) #X, y


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lamda, epochs):
    #  YOUR CODE HERE
    #m = len(y)
    # Add bias term
    #X = np.c_[np.ones((m, 1)), X]  
    w = np.zeros(X.shape[1])  # Initialize weights
    cost_history = [] # List to store cost values for plotting cost(w) vs epochs
    for epoch in range(epochs):
        gradient = compute_gradient(X, y, w)
        w -= lamda * gradient
        cost = compute_cost(X, y, w)
        cost_history.append(cost)
    return w, cost_history


# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w):
    #  YOUR CODE HERE
    predictions = X.dot(w)
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    return rmse 


# Compute objective (cost) function.
def compute_cost(X, y, w):
    #  YOUR CODE HERE
    # prediciton of y_tahmin
    y_tahmin = X.dot(w)
    m = len(y)
    # compute cost
    cost = (1 / (2 * m)) * np.sum((y_tahmin - y) ** 2) #0.5 * np.mean((y_tahmin - y) ** 2)
    return cost 


# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):
    #  YOUR CODE HERE
    m = len(y)
    #grad = np.zeros(w.shape)
    # Predict y_tahmin
    y_tahmin = X.dot(w)
    # Compute gradient
    grad = (1 / m) * X.T.dot(y_tahmin - y)
    return grad

##======================= Main program =======================##

# Main program (değiştirilmiş kısım)
Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

# Standardizing the features
mean, std = scaling.mean_std(Xtrain)
Xtrain_std = scaling.standardize(Xtrain, mean, std)
Xtest_std = scaling.standardize(Xtest, mean, std)

# Add the bias term to the standardized training data
Xtrain_std_b = np.c_[np.ones((len(Xtrain_std), 1)), Xtrain_std]

# Training the model using gradient descent
learning_rate = 0.1
epochs = 500
w, cost_history = train(Xtrain_std_b, ttrain, learning_rate, epochs)


# Print the parameters  print("Trained parameters:", w)
print("Trained parameters: [", ", ".join(f"{param:.2f}" for param in w), "]")

# Calculate and print RMSE for training and test sets
train_rmse = compute_rmse(np.c_[np.ones((len(Xtrain_std), 1)), Xtrain_std], ttrain, w)
test_rmse = compute_rmse(np.c_[np.ones((len(Xtest_std), 1)), Xtest_std], ttest, w)

print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Calculate parameters using the normal equation
X_b = np.c_[np.ones((len(Xtrain), 1)), Xtrain]  

# Calculate parameters using the normal equation
w_normal_eq = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(ttrain)

print("Parameters from Normal Equation:", w_normal_eq)

# Plot J(w) vs. number of epochs
plt.figure()
plt.plot(cost_history)
plt.title("Cost function J(w) vs. Number of epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost J(w)")
plt.show()

# Plot the results
plt.figure()
plt.scatter(Xtrain_std, ttrain, color='blue', label='Train data', marker='o')
plt.scatter(Xtest_std, ttest, color='green', label='Test data', marker='x')

# Generate a line for predictions based on standardized data
X_line_std = np.linspace(min(Xtrain_std), max(Xtrain_std), 100).reshape(-1, 1)  # Generate 100 points in standardized range
X_line_b = np.c_[np.ones((X_line_std.shape[0], 1)), X_line_std]  # Add bias term
y_line = X_line_b.dot(w)  # Predictions using standardized X_line and trained weights

# Plot the regression line
plt.plot(X_line_std, y_line, color='red', label='Linear regression line')
plt.title("House Prices vs Floor Size (Standardized)")
plt.xlabel("Floor Size")
plt.ylabel("House Price")
plt.legend()
plt.show()