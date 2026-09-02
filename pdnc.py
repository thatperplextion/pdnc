# 1. import libraries
import numpy as np

# 2. XOR dataset
# [0, 0] -> 0
# [0, 1] -> 1
# [1, 0] -> 1
# [1, 1] -> 0

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# 3. Network architecture:
# Input  = 2 neurons
# Hidden = 4 neurons
# Output = 1 neuron

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# 4. INITIALIZE PARAMETERS
np.random.seed(42)

# Hidden layer parameters
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.5
b1 = np.zeros((1, hidden_neurons))

# Output layer parameters
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.5
b2 = np.zeros((1, output_neurons))


# 5. ACTIVATION FUNCTIONS

def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# 6. LOSS FUNCTION

def binary_cross_entropy(Y, Y_hat):

    # Small value to avoid log(0) because log(0) is undefined and will result in NaN values.
    epsilon = 1e-8

    loss = -np.mean(
        Y * np.log(Y_hat + epsilon)
        +
        (1 - Y) * np.log(1 - Y_hat + epsilon)
    )

    return loss

# 7. FORWARD PROPAGATION

def forward_propagation(X, W1, b1, W2, b2):

    # Hidden Layer Z1 = XW1 + b1
    Z1 = np.dot(X, W1) + b1

    # Activation - ReLU(Z1)
    A1 = relu(Z1)

    # Output Layer Z2 = A1W2 + b2
    Z2 = np.dot(A1, W2) + b2

    # Y_hat (output predictions) = sigmoid(Z2)
    Y_hat = sigmoid(Z2)

    return Z1, A1, Z2, Y_hat



# 8. BACKPROPAGATION

def backward_propagation(X, Y, Z1, A1, Z2, Y_hat, W2):

    m = X.shape[0]

    # STEP 1:
    # Derivative of BCE loss with respect to Z2
    # For sigmoid + binary cross entropy:
    # dL/dZ2 = Y_hat - Y
    
    dZ2 = Y_hat - Y

    # STEP 2:
    # Gradient of W2
    # Z2 = A1 W2 + b2
    # Therefore:
    # dL/dW2 = A1.T @ dZ2
    
    dW2 = np.dot(A1.T, dZ2) / m

    # STEP 3:
    # Gradient of b2
    
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # STEP 4:
    # Propagate gradient backward to A1
    # dL/dA1 = dL/dZ2 * dZ2/dA1
    # Since:
    # Z2 = A1 W2
    # dL/dA1 = dZ2 @ W2.T
    
    dA1 = np.dot(dZ2, W2.T)

    # STEP 5:
    # Backpropagate through ReLU
    # A1 = ReLU(Z1)
    # dL/dZ1 =
    # dL/dA1 * dA1/dZ1
    
    dZ1 = dA1 * relu_derivative(Z1)

    # STEP 6:
    # Gradient of W1
    # Z1 = X W1 + b1
    # dL/dW1 = X.T @ dZ1
    
    dW1 = np.dot(X.T, dZ1) / m

    # STEP 7:
    # Gradient of b1
    
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# 9. GRADIENT DESCENT

learning_rate = 0.5
epochs = 10000


for epoch in range(epochs):

    # FORWARD PROPAGATION
    
    Z1, A1, Z2, Y_hat = forward_propagation(
        X, W1, b1, W2, b2
    )

    # LOSS CALCULATION
    
    loss = binary_cross_entropy(Y, Y_hat)

    # BACKWARD PROPAGATION
    
    dW1, db1, dW2, db2 = backward_propagation(
        X,
        Y,
        Z1,
        A1,
        Z2,
        Y_hat,
        W2
    )

    # GRADIENT DESCENT
    # New weight = Old weight - learning_rate * gradient
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # Print loss
    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", loss)

# 10. FINAL PREDICTIONS

_, _, _, predictions = forward_propagation(
    X, W1, b1, W2, b2
)

print("\nFinal probabilities:")
print(predictions)

print("\nFinal predictions:")

binary_predictions = (predictions >= 0.5).astype(int)

print(binary_predictions)

print("\nActual values:")
print(Y)