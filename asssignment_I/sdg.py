import numpy as np
import matplotlib.pyplot as plt
from LoadClaus import load_training_dataset, load_test_dataset

# Step 1: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 2: Gradient of the loss function
def compute_gradient(X, y, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y)) / y.size
    return gradient

# Step 3: Update weights
def update_weights(X, y, weights, lr):
    gradient = compute_gradient(X, y, weights)
    weights -= lr * gradient
    return weights

# Step 4: Training the model
def train(X, y, X_test, y_test, lr=0.01, num_iter=100000):
    weights = np.zeros(X.shape[1])
    train_errors = []
    test_errors = []

    for _ in range(num_iter):
        # Shuffle the data
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

        for i in range(X.shape[0]):
            weights = update_weights(X[i:], y[i:i+1], weights, lr)

            # Calculate and store the errors after each update
            train_pred = predict(X, weights)
            test_pred = predict(X_test, weights)
            train_errors.append(mean_squared_error(y, train_pred))
            test_errors.append(mean_squared_error(y_test, test_pred))

    return weights, train_errors, test_errors

# Step 5: Predicting the class label
def predict(X, weights):
    z = np.dot(X, weights)
    return [1 if i > 0.5 else 0 for i in sigmoid(z)]


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

X_train, y_train = load_training_dataset()
X_test, y_test = load_test_dataset()

# Train the model and get the errors
weights, train_errors, test_errors = train(X_train, y_train, X_test, y_test)

# Plot the errors
plt.figure(figsize=(12, 6))
plt.plot(train_errors, label='Train Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Number of iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()