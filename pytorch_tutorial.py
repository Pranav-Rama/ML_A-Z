import torch
import numpy as np

# Input (temp, rainfall, humidity)
features = np.array([[73, 67, 43], 
                     [91, 88,64], 
                     [87, 134, 58], 
                     [102, 43, 37], 
                     [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
dep_var = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Convert to tensors:
features = torch.from_numpy(features)
dep_var = torch.from_numpy(dep_var)

print(features)
print(features.dtype)
print(dep_var)
print(dep_var.dtype)

# Initiate random weights and biases:
weights = torch.randn(2, 3, requires_grad=True)
bias = torch.randn(2, requires_grad=True)

print('Random weights: ', weights)
print('Random bias: ', bias)

# The model is define by the equation: w1*x1 + w2*x2 + w3*x3 + b1 = prediction 
def regressor(x):
    return x @ weights.t() + bias

# Generate predictions: 
pred = regressor(features)
print('Predictions: ', pred)

# Compare with actual values: 
print('Target: ', dep_var)

# Calculate MSE (Loss Function): 
# # 1. Calculate difference between prediction and target:
# diff = pred - dep_var
# print('Difference: ', diff)

def MSE(t1, t2): 
    diff = t1 - t2
    return torch.sum(diff*diff) / diff.numel() #numel is the number of elements 

# Compute the loss
loss = MSE(pred, dep_var)
print('Loss: ', loss)

# Compute Gradients for loss:
loss.backward()

# Gradient for weights:
print('Weights: ', weights)
print('Weights gradient: ', weights.grad)

# Gradient of biases: 
print('Bias: ', bias)
print('Bias gradients: ', bias.grad)

# Reset gradient values to zero: 
weights.grad.zero_()
bias.grad.zero_()
print(weights.grad)
print(bias.grad)

## Training Model; Adjusting weights and biases using gradient decent:
# 1) Generate Predictions
# 2) Calculate the loss
# 3) Compute gradients for weights and biases
# 4) Adjust weights by substaction a small value propotional tio the gradient 
# 5) Reset the Gradients to zero

# Generate predictions: 
pred = regressor(features)
print('Predictions: ', pred)

# Compute the loss
loss = MSE(pred, dep_var)
print('Loss: ', loss)

# Compute Gradients for loss:
loss.backward()
print(weights.grad)
print(bias.grad)

# Adjust weights then reset to zero:
with torch.no_grad():
    weights -= weights.grad * 1e-5
    bias -= bias.grad * 1e-5
    weights.grad.zero_()
    bias.grad.zero_()

print('New weights: ', weights)
print('New bias: ', bias)

# Calculate loss:
pred = regressor(features)
loss = MSE(pred, dep_var)
print('Loss: ', loss)