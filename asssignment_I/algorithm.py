import numpy as np
from LoadClaus import load_training_dataset, load_test_dataset
import matplotlib.pyplot as plt



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(y, z):
    return y * np.log(1 + np.exp(-z)) + (1 - y) * np.log(1 + np.exp(z))


def predict(X, theta):
    print(X.shape)
    print(theta.shape)
    z = np.dot(X, theta)
    print(z.size)

    predictions = np.zeros(z.size)

    for i in range(z.size):
        predictions[i] = 1 if sigmoid(z[i]) > 0.5 else 0
    return predictions

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def sgd(X, y, theta, learning_rate, num_iterations):
    m = len(y)

    print(m)
    losses = []


    for i in range(num_iterations):
        # print(i)
        rand_ind = np.random.randint(0, m)
        X_i = X[rand_ind,:].reshape(1, X.shape[1])
        y_i = y[rand_ind].reshape(1, 1)


        gradient =  1/m * np.dot(X_i.T, (y_i - sigmoid(np.dot(X_i, theta))))
        theta -= learning_rate * gradient
        
        losses.append(loss(y_i, sigmoid(np.dot(X_i, theta)))[0][0])


    return theta, losses



def accuracy(y, y_i):
    print(y.shape)
    print(y_i.shape)
    accuracy = np.sum(y == y_i) / len(y)
    return accuracy

def plot_decision_boundary(X, y, w, b):
    
    # X --> Inputs
    # w --> weights
    # b --> bias
    
    # The Line is y=mx+c
    # So, Equate mx+c = w.X + b
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -w[0]/w[1]
    c = -b/w[1]
    x2 = m*x1 + c
    
    # Plotting
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0], X[:, 1], "g^")
    plt.plot(X[:, 0], X[:, 1], "bs")
    plt.xlim([-2, 2])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')    
    plt.plot(x1, x2, 'y-')


# def errorA(X, y, theta):
#     m = len(y)
#     predictions = sigmoid(np.dot(X, theta))
#     # return 1/m * np.sum((y - predictions) ** 2)
#     return np.sum(np.sign(np.dot(X, theta)) == y) / m







def train(X_train, Y_train):

    theta = np.zeros([X_train.shape[1], 1])
    learning_rate = 0.001
    num_iterations = 1000
    theta, losses = sgd(X_train, Y_train, theta, learning_rate, num_iterations)


   # Plot the loss history
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')
    plt.show()


    return theta, losses
    # Initialize the weights

    # Training error: [[1.16666667]]
    # Test error: [[1.16666667]]

    # Training error: [[1.19543112]]
    # Test error: [[1.19586358]]



    # Calculate the training and test errors


#     print(f'Training error: {errorA(X_train, y_train, theta)}')
#     print(f'Test error: {errorA(X_test, y_test, theta)}')

# #   Training error: [[0.80741032]]
# #   Test error: [[0.80741032]]


#     # Define the learning rate and the number of iterations
#     learning_rate = 0.01
#     num_iterations = 100000

#     # Train the model using SGD
#     theta = sgd(X_train, y_train, theta, learning_rate, num_iterations)



    # print(f'Prediction: {pred}')
    # print(f'Actual: {actual}')
    # evaluate

    # Load the test data

    

    # Calculate the training and test errors
    # train_loss = total_loss(X_train, y_train, theta)
    # test_loss = total_loss(X_test, y_test, theta)

    #Training error: 0.25080820024748474
    # Test error: 0.2508155767497876

    # print(f'Training error: {errorA(X_train, y_train, theta)}')
    # print(f'Test error: {errorA(X_test, y_test, theta)}')




    # train_error = loss(y_train, sigmoid(np.dot(X_train, theta)))
    # test_error = loss(y_test, sigmoid(np.dot(X_test, theta)))

   


    


    # Print the training and test errors

    # # Plot the cost history
    # plt.plot()
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss history')
    # plt.show()

    # print(theta.size)

    # Reshape the weights into a 28x28 image
    weights_image = theta.reshape(28, 28)

    # Plot the weights image
    plt.imshow(weights_image, cmap='gray')
    plt.title('Weights')
    plt.show()

    # Plot the weights image
    plt.imshow(weights_image, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Weights Heatmap')
    plt.show()


def main():
    X_train, Y_train = load_training_dataset()
    X_test, y_test = load_test_dataset()

    theta, losses = train(X_train, Y_train)

 

    acc = accuracy(Y_train, predict(X_train, theta).reshape(Y_train.shape))
    print(f'Accuracy: {acc}')



main()