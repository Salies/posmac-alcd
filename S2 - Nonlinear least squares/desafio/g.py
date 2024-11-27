from scipy.optimize import least_squares
import numpy as np

# Define sigmoid function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

# Define the neural network function
def f_hat(x, theta):
    z1 = theta[1] * x[0] + theta[2] * x[1] + theta[3]
    z2 = theta[5] * x[0] + theta[6] * x[1] + theta[7]
    z3 = theta[9] * x[0] + theta[10] * x[1] + theta[11]
    return (theta[0] * sigmoid(z1) + theta[4] * sigmoid(z2) +
            theta[8] * sigmoid(z3) + theta[12])

# Residuals
def residuals(theta, X, Y):
    return np.array([f_hat(X[i], theta) - Y[i] for i in range(len(Y))])

# Generate data
np.random.seed(0)
X = np.random.uniform(-1, 1, (200, 2))
Y = X[:, 0] * X[:, 1]

# Initial guess
theta_init = np.random.randn(13)

# Fit the model
result = least_squares(residuals, theta_init, args=(X, Y))

print(result)