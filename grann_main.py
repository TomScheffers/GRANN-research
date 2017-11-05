import networkx as nx
import numpy as np
import builtins
from strategic_functions import new_edge
from iteration_functions import iterate, calculate_preds
from generate_dataset import generate_dataset
from generate_grid import make_graph
from plotting_functions import plot

data_size = 100
output_node = 1000 #this is also the max amount of nodes you can have.
function_type = "Sinus"
X_train, X_test, y_train, y_test = generate_dataset(data_size, function_type)
iterations = 100
lr = 0.005
L1 = 0
L2 = 1e-5
n = int(X_train.shape[0]/10)
builtins.activation_type = "SELU" #Sigmoid, SELU, ELU, RELU
builtins.G = nx.Graph()
hidden_nodes, G = make_graph(input_nodes=1, hidden_nodes=5, hidden_connections=2, steps_back=3, output_connections=1, output_node)

value_matrix, predictions, total_errors = iterate(iterations, X_train, y_train, hidden_nodes, L1, L2, lr)
new_edge(value_matrix, hidden_nodes, n)
value_matrix, predictions, total_errors = iterate(iterations, X_train, y_train, hidden_nodes, L1, L2, lr)
new_node()

mse, y_preds = calculate_preds(X_test, y_test, hidden_nodes)
plot(X_train, y_train, X_test, y_test, y_preds, data_size, function_type, hidden_nodes)
