import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.sum((y_pred - y_true) ** 2) / (2 * m)

    def backward(self, X, y_true):
        m = X.shape[0]
        dA2 = (self.A2 - y_true) / m
        dZ2 = dA2 * self.A2 * (1 - self.A2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return self.grads

    def update(self, lr=0.1):
        self.W1 -= lr * self.grads["dW1"]
        self.b1 -= lr * self.grads["db1"]
        self.W2 -= lr * self.grads["dW2"]
        self.b2 -= lr * self.grads["db2"]