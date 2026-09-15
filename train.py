import numpy as np
from sklearn.datasets import load_digits
from model import SimpleNN


digits = load_digits(n_class=2)
X = digits.data / 16.0                          
y = digits.target.reshape(-1, 1).astype(float)

net = SimpleNN(input_size=X.shape[1], hidden_size=16, output_size=1)

losses = []
for epoch in range(200):
    y_pred = net.forward(X)
    loss = net.compute_loss(y_pred, y)
    net.backward(X, y)
    net.update(lr=0.5)
    losses.append(loss)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"Final loss: {losses[-1]:.4f}")