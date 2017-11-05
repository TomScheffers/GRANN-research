import networkx as nx
import numpy as np
import builtins
from strategic_functions import new_edge, residual_patching
from iteration_functions import iterate, calculate_preds
from generate_dataset import generate_dataset
from generate_grid import make_graph
from plotting_functions import plot

data_size = 100
output_node = 1000 #this is also the max amount of nodes you can have.
function_type = "Sinus"
X_train, X_test, y_train, y_test = generate_dataset(data_size, function_type)
iterations = 200
lr = 0.005
L1 = 0
L2 = 0 #1e-5
n = int(X_train.shape[0]/10)
builtins.activation_type = "SELU" #Sigmoid, SELU, ELU, RELU
builtins.G = nx.Graph()
try:
    input_dim = X_train.shape[1]
except:
    input_dim = 1
hidden_nodes = make_graph(output_node, input_dim, hidden_nodes=5, hidden_connections=2, steps_back=3, output_connections=1)

#value_matrix, predictions, total_errors = iterate(iterations, X_train, y_train, hidden_nodes, L1, L2, lr, output_node)
#new_edge(value_matrix, hidden_nodes, n, output_node)
value_matrix, residuals, total_errors = iterate(iterations, X_train, y_train, hidden_nodes, L1, L2, lr, output_node, residual = False)
hidden_nodes = residual_patching(4, iterations * 5, residuals, output_node, input_dim, L1, L2, lr, hidden_nodes, X_train)
value_matrix, predictions, total_errors = iterate(iterations, X_train, y_train, hidden_nodes, L1, L2, lr, output_node, residual = False)

mse, y_preds = calculate_preds(X_test, y_test, hidden_nodes, output_node)
plot(X_train, y_train, X_test, y_test, y_preds, data_size, function_type, hidden_nodes)
