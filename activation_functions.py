import numpy as np

def activation_func(x):
    global activation_type
    if activation_type == "ELU":
        alpha = 1
        if x >= 0:
            return x
        else:
            return alpha * np.exp(x) - alpha
    elif activation_type == "RELU":
        if x >= 0:
            return x
        else:
            return 0
    elif activation_type == "SELU":
        lamb = 1.0507
        alpha = 1.6732
        if x > 0:
            return lamb * x
        else:
            return lamb * (alpha * np.exp(x) - alpha)
    elif activation_type == "Sigmoid":
        return 1 / (1 + np.exp(-x))

def activation_der(x):
    global activation_type
    if activation_type == "ELU":
        alpha = 2
        if x >= 0:
            return 1
        else:
            return activation_func(x) + alpha
    elif activation_type == "RELU":
        if x >= 0:
            return 1
        else:
            return 0
    elif activation_type == "SELU":
        lamb = 1.0507
        alpha = 1.6732
        if x > 0:
            return lamb
        else:
            return lamb * alpha * np.exp(x)
    elif activation_type == "Sigmoid":
        return activation_func(x) * (1 - activation_func(x))
