import numpy as np
import torch
from model import SimpleNN

np.random.seed(0)


X = np.random.randn(8, 4)
y = np.random.randint(0, 2, size=(8, 1)).astype(float)


net = SimpleNN(input_size=4, hidden_size=5, output_size=1)
y_pred = net.forward(X)
loss = net.compute_loss(y_pred, y)
grads = net.backward(X, y)


X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
W1_t = torch.tensor(net.W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(net.b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(net.W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(net.b2, dtype=torch.float32, requires_grad=True)

Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
A2_t = torch.sigmoid(Z2_t)

loss_t = ((A2_t - y_t) ** 2).sum() / (2 * X_t.shape[0]) 
loss_t.backward()

def compare(name, mine, torch_grad):
    diff = np.abs(mine - torch_grad.detach().numpy()).max()
    status = "PASS" if diff < 1e-5 else "FAIL"
    print(f"{name}: max diff = {diff:.2e}  [{status}]")

compare("dW1", grads["dW1"], W1_t.grad)
compare("db1", grads["db1"], b1_t.grad)
compare("dW2", grads["dW2"], W2_t.grad)
compare("db2", grads["db2"], b2_t.grad)