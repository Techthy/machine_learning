
import numpy as np
from LoadClaus import load_training_dataset, load_test_dataset
import matplotlib.pyplot as plt



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def grad_logistic(x, y, theta, N):
    return 1/N * (x.T).dot(-y  + y * sigmoid(np.multiply(y, x.dot(theta))))

def grad_ols(x, y, theta, N):
    return 2/N * ((x.T).dot(x.dot(theta) - y))

def logistic_loss(x, y, theta, N):
    h_theta = sigmoid(x.dot(theta))
    cost = -1/N * np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
    return cost

def logistic_sgd_with_loss(X, y, learning_rate, num_iterations, X_test, y_test):
    m = len(y)
    theta = np.zeros([X.shape[1], 1])
    train_losses = []
    test_losses = []
    print("starting training with measuring errors...")

    for i in range(num_iterations):
        print(f' iteration: {i}', end='\r')
        rand_ind = np.random.randint(0, m)
        X_i = X[rand_ind,:].reshape(1, X.shape[1])
        y_i = y[rand_ind].reshape(1, 1)

        gradient = grad_logistic(X_i, y_i, theta, m)
        theta -= learning_rate * gradient
        
        train_losses.append(logistic_loss(X, y, theta, m))
        test_losses.append(logistic_loss(X_test, y_test, theta, m))

    return theta, train_losses, test_losses



def sdg(X, y, learning_rate, num_iterations, grad_func=grad_logistic):
    m = len(y)
    theta = np.zeros([X.shape[1], 1])

    for i in range(num_iterations):
        print(f'training at iteration: {i}', end='\r')
        rand_ind = np.random.randint(0, m)
        X_i = X[rand_ind,:].reshape(1, X.shape[1])
        y_i = y[rand_ind].reshape(1, 1)

        gradient = grad_func(X_i, y_i, theta, m)
        theta -= learning_rate * gradient
        
    print("training finished")

    return theta

def perceptron(X, y, learning_rate, num_iterations):
    b = 0
    m = len(y)
    w = np.zeros(X.shape[1])

    for i in range(num_iterations):
        print(f'training at iteration: {i}', end='\r')
        rand_ind = np.random.randint(0, m)
        X_i = X[rand_ind,:]
        y_i = y[rand_ind]


        z = np.dot(X_i, w) + b

        if y_i * z <= 0:
            w = w + learning_rate * y_i * X_i
            b = b + learning_rate * y_i

    print("training finished")
    
    return b, w


def prediction(x, theta):
    y_pred = sigmoid(x.dot(theta)) 
    y_class = [1 if i >= 0.5 else -1 for i in y_pred]
    return np.array(y_class)


def ols_predict(X, theta):
    print(np.dot(X, theta).shape)   
    return np.dot(X, theta)

def plot_train_error(losses, title , learning_rate):
    plt.plot(losses)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss History with Learning Rate {learning_rate}')
    plt.show()

def evaluate(X, y, theta):
    y_pred = prediction(X, theta)
    acc = np.sum(y_pred.reshape(y.shape) != y) / len(y)
    return acc

def evaluate1( X_test, y_test, theta, bias):
    predictions = []
    
    for i in range(X_test.shape[0]):
        activation = np.dot(X_test[i], theta) + bias
        y_pred = 1 if activation >= 0 else -1
        predictions.append(y_pred)

    errors = np.sum(np.array(predictions).reshape(y_test.shape) != y_test)
    return errors / len(y_test)
    

def train_and_evaluate_logistic(X_train, Y_train, X_test, Y_test):
    learning_rate = 0.01
    iterations = [10, 100, 1000, 10000, 100000]


    for i in iterations:
        theta = sdg(X_train, Y_train, learning_rate, i, grad_logistic)
        zero_one_err_train = evaluate(X_train, Y_train, theta)
        print(f'Logistic Regression Empirical Error (0-1 loss) for training set at iteration {i}: {zero_one_err_train}')
        zero_one_err_test = evaluate(X_test, Y_test, theta)
        print(f'Logistic Regression 0-1 Error (0-1 loss) for test set at iteration {i}: {zero_one_err_test}')
        weights_image = theta.reshape(28, 28)
        plt.imshow(weights_image, cmap='gray')
        plt.title('Weights')
        plt.show()


def train_and_evaluate_OLS(X_train, Y_train, X_test, Y_test):
    learning_rate = 0.03
    iterations = [10, 100, 1000, 10000, 100000]


    for i in iterations:
        theta = sdg(X_train, Y_train, learning_rate, i, grad_ols)
        zero_one_err_train = evaluate(X_train, Y_train, theta)
        print(f'OLS Empirical Error (0-1 loss) for training set at iteration {i}: {zero_one_err_train}')
        zero_one_err_test = evaluate(X_test, Y_test, theta)
        print(f'OLS Test 0-1 Error (0-1 loss) for test set at iteration {i}: {zero_one_err_test}')
        weights_image = theta.reshape(28, 28)
        plt.imshow(weights_image, cmap='gray')
        plt.title('Weights')
        plt.show()




def train_and_evalutate_perceptron(X_train, Y_train, X_test, Y_test):
    learning_rate = 0.01
    iterations = [10, 100, 1000, 10000, 100000]


    for i in iterations:
        b, theta = perceptron(X_train, Y_train, learning_rate, i)
        zero_one_err_train = evaluate1(X_train, Y_train, theta, b)
        print(f'Perceptron Empirical Error (0-1 loss) for training set at iteration {i}: {zero_one_err_train}')
        zero_one_err_test = evaluate1(X_test, Y_test, theta, b)
        print(f'Perceptron Test 0-1 Error (0-1 loss) for test set at iteration {i}: {zero_one_err_test}')
        weights_image = theta.reshape(28, 28)
        plt.imshow(weights_image, cmap='gray')
        plt.title('Weights')
        plt.show()


def logistic_reg_diff_learning_rates(X_train, Y_train, X_test, Y_test, learning_rates=[1, 0.1, 0.01, 0.001]):
    for rate in learning_rates:
        theta, train_losses, test_losses = logistic_sgd_with_loss(X_train, Y_train, rate, 10000, X_test, Y_test)
        plot_train_error(train_losses, 'Train', rate)
        plot_train_error(test_losses, 'Test', rate)
        zero_one_err_train = evaluate(X_train, Y_train, theta)
        print(f'Empirical Error (0-1 loss) for training set with learning rate {rate}: {zero_one_err_train}')
        zero_one_err_test = evaluate(X_test, Y_test, theta)
        print(f'Test 0-1 Error (0-1 loss) for test set with learning rate {rate}: {zero_one_err_test}')





def main():
    X_train, Y_train = load_training_dataset()
    X_test, Y_test = load_test_dataset()


    # logistic_reg_diff_learning_rates(X_train, Y_train, X_test, Y_test)
    train_and_evaluate_logistic(X_train, Y_train, X_test, Y_test)
    train_and_evaluate_OLS(X_train, Y_train, X_test, Y_test)
    train_and_evalutate_perceptron(X_train, Y_train, X_test, Y_test)


main()