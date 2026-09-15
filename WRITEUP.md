

What I built

I built a small feedforward neural network from scratch in NumPy — one hidden layer, no autograd, no `.backward()` anywhere. The architecture is:


Input (n features) → Linear → ReLU → Linear → Sigmoid → Output (1 value)


I trained it on the `load_digits` dataset from scikit-learn, restricted to two classes (0 and 1), which turns it into a binary classification problem. 

For the loss function, I went with **quadratic loss (mean squared error)** . I'll be upfront about why: I'd already watched 3Blue1Brown's backprop videos using quadratic loss, so deriving gradients for a loss function I'd already seen worked through once meant I could actually follow my own math instead of learning a new derivation cold. 

Architecture, precisely

 Input layer: 64 neurons (8×8 pixel images from `load_digits`, flattened)
 Hidden layer: 16 neurons, ReLU activation
 Output layer: 1 neuron, sigmoid activation

Weights are initialized as small random values (`* 0.01`) so activations don't start too large; biases start at zero.

 The math: deriving the backward pass

Forward pass:

$$Z_1 = XW_1 + b_1, \quad A_1 = \text{ReLU}(Z_1), \quad Z_2 = A_1W_2 + b_2, \quad A_2 = \sigma(Z_2)$$

Loss:

$$L = \frac{1}{2m}\sum (A_2 - y)^2$$

**1) gradient of the loss w.r.t. the prediction.** Differentiating the quadratic loss with respect to $A_2$, the exponent's 2 cancels the $\frac{1}{2}$:

$$\frac{\partial L}{\partial A_2} = \frac{1}{m}(A_2 - y)$$

**2) — derivative of the output (A2) with respect to Z2 (sigmoid).** Sigmoid's derivative can be written entirely in terms of its own output.

$$\frac{\partial A_2}{\partial Z_2} = A_2(1 - A_2) \quad \Rightarrow \quad \frac{\partial L}{\partial Z_2} = \frac{1}{m}(A_2-y) \cdot A_2 \cdot (1-A_2)$$

This is why `self.A2` gets stored during the forward pass — the backward pass needs it directly, no recomputation required.

**3) — gradients for the second layer.** Since $Z_2 = A_1 W_2 + b_2$:

$$\frac{\partial L}{\partial W_2} = A_1^T \cdot \frac{\partial L}{\partial Z_2}, \qquad \frac{\partial L}{\partial b_2} = \sum_{\text{samples}} \frac{\partial L}{\partial Z_2}$$

**Step 4 — pushing the gradient back into the hidden layer.** From $Z_2 = A_1 W_2 + b_2$:

$$\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial Z_2} \cdot W_2^T$$

**Step 5 — through ReLU.** ReLU's derivative is 1 where the pre-activation was positive, 0 otherwise:

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{1}[Z_1 > 0]$$

**Step 6 — gradients for the first layer.** Same pattern as Step 3, using $X$ instead of $A_1$:

$$\frac{\partial L}{\partial W_1} = X^T \cdot \frac{\partial L}{\partial Z_1}, \qquad \frac{\partial L}{\partial b_1} = \sum_{\text{samples}} \frac{\partial L}{\partial Z_1}$$

Every gradient's shape matches the parameter it belongs to — $\partial L/\partial W_1$ is `(input_size, hidden_size)`, same as $W_1$ itself, and so on down the line. That property gave me so many errors because I didn't think that much when to take transpose or not .

## Gradient checking

To confirm the manual derivation above was actually correct, I built the same network in PyTorch, copied over my exact weight values, ran `.backward()`, and compared PyTorch's autograd gradients against mine. All four matched to within floating-point noise:


dW1: max diff = 3.15e-11  [PASS]
db1: max diff = 7.17e-12  [PASS]
dW2: max diff = 1.04e-10  [PASS]
db2: max diff = 1.32e-09  [PASS]


These differences are on the order of $10^{-9}$ to $10^{-11}$, which is the kind of noise you'd expect from floating-point arithmetic doing the same computation two different ways 

## Training results

I trained the network for 200 epochs on the digits dataset (normalized to [0,1]) with a learning rate of 0.5. The loss dropped steadily across training — starting around 0.2 and falling to well under 0.05 by the final epoch, with no spikes or instability. This confirms the network is actually learning, not just running without crashing.

## Mistakes I made, and how I found them

**A dtype mismatch in the gradient-check script.** When building the PyTorch comparison network, I hit:


RuntimeError: expected m1 and m2 to have the same dtype, but got: float != double


The why: NumPy's `np.random.randn(...)` defaults to `float64`, but I'd explicitly cast my input tensors to `float32` without doing the same for the weight tensors copied over from my NumPy network. PyTorch refuses to matrix-multiply tensors of two different precisions. The fix was simple, pass `dtype=torch.float32` when converting every NumPy array to a PyTorch tensor, not just the inputs. 

**The near-miss with transposes.** While writing the backward pass, I spent time thinking through exactly where each `.T` needed to go — e.g. `dW2 = self.A1.T @ dZ2` rather than `self.A1 @ dZ2`. Which gave errors when multiplying the matrices since it didn't have the correct order 


## Summary

Everything here works reasonably : the forward pass produces sensible outputs, the manually-derived backward pass matches PyTorch's autograd to floating-point precision, and the network's loss decreases over training on real data. The two bugs I hit along the way — the dtype mismatch and the transpose consideration — were both things that i encountered and resolved promptly.Overall I believe this task was a good learning experience and an introduction to the world of neural networks.