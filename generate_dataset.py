import numpy as np
from sklearn.model_selection import train_test_split

def generate_dataset(data_size, function_type):
    X = np.linspace(-1, 1, data_size)
    e = np.random.normal(0, scale=0.2, size=data_size)
    if function_type == "Sinus":
        y = np.sin(6.28*X) + e
    elif function_type == "Quadratic":
        y = np.power(X,2) + e
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test
