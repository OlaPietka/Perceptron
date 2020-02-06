import csv
import numpy
import random


def sum_weighted(X, W):
    sum_w = 0

    for i in range(len(X)):  
        for j in range(len(W)): 
            sum_w += W[j] * X[i][j]

    return sum_w


def f(x):
    """
    Hyperbolic tangent
    """
    return numpy.tanh(x)


def fprim(x):
    """
    Derivative to the hyperbolic tangent
    """
    return 1 - numpy.tanh(x)*numpy.tanh(x)


def read_input_data(filename):
    '''
    Reads input data (train / test) from a CSV file.
    Input:
        filename - CSV file name (string)
    CSV file format:
        input1, input2, ..., output
                        ...
                        ...
    Returns:
        Nin - number of inputs of the perceptron (int)
        X - input training data (list)
        Y - output (expected) training data (list)
    '''

    Nin = 2
    X = []
    Y = []

    file = open(filename, "r")
    data = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)  

    for row in data:
        X.append(row[0:Nin]) 
        Y.append(row[Nin])
    file.close()

    return Nin, X, Y


def initialize_weights(Nin):
    '''
    Initialize weights with a random numbers from range [0,1).
    Input:
        Nin - number of inputs of the perceptron (int)
    Output:
        Randomly initialized weights (list of Nin size)
    '''

    w = []
    for i in range(Nin):
        w.append(random.random()) 

    return w


def train(epochs, X, Y, weights, eta):
    '''
    Trains the simple perceptron using the gradient method.
    Plots the RMSE.
    Inputs:
        epochs - number of training iterations (int > 0)
        X - training (input) vector (list)
        Y - training (output) vector (list)
        weights - initial weights (list)
        eta - learning rate (0-1]
    Returns:
        weights - optimized weights (list)
    '''

    # For each epoch:
    #   For each training data (pair X,Y):
    #       Calculate output Yout
    #       Use Yout to calculate error
    #       Adjust weights using the gradient formula
    #       Calculate ans store the RMSE (root mean squared error)
    # Plot the RMSE(epoch)

    for epoch in range(epochs):
        print("Epoch no. {}".format(epoch))

        error = 0
        sumWeighted = float(sum_weighted(X, weights))

        for i in range(len(X)): 
            Yout = f(sumWeighted)
            error += pow(Y[i] - Yout, 2)

            for j in range(Nin):
                weights[j] += eta*fprim(sumWeighted)*(Y[i] - Yout)*X[i][j]

    return weights


def test(filename, weights):
    '''
    Test ot the trained perceptron by propagating the test data.
    Input:
        filename - CSV file name (string)
        weights - trained weights (list)
    CSV file format:
        input1, input2, ..., expected_output
                        ...
                        ...
    Returns:
        Y - output testing results (list)
        Yexpected - output expected results (list)

    '''

    Y = []
    Nin, Xtest, Yexpected = read_input_data(filename)

    for i in range(len(Xtest)): 
        sumWeighted = 0
        for j in range(Nin): 
            sumWeighted += weights[j]*Xtest[i][j]

        Y.append(f(sumWeighted))

    return Y, Yexpected


if __name__ == '__main__':
    '''
     Simple perceptron

                 Yout
                  ^
                  |
                  O
                / | \         weights: weights[]
              Nin inputs
    '''

    # Get the train data
    Nin, Xtrain, Ytrain = read_input_data("data/train_data.csv")

    # Initialize weights
    weights = initialize_weights(Nin)

    c = weights
    # Train of the perceptron
    epochs = 10000  
    eta = 0.01  
    weights = train(epochs, Xtrain, Ytrain, weights, eta)

    # Test of the perceptron with the trained weights
    Yout, Yexpected = test("data/test_data.csv", weights)

    print(c)
    for i in range(len(Yout)):
        print("Results:", Yout[i])
        print("Expected results:", Yexpected[i])

