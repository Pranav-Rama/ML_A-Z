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

# Stopped at 53.17


