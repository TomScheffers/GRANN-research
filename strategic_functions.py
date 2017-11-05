import numpy as np
import networkx as nx
from generate_grid import generate_node
from iteration_functions import iterate
import matplotlib.pyplot as plt

def new_edge(value_matrix, hidden_nodes, n, output_node):
    value_matrix = value_matrix[np.argsort(value_matrix[:,0])]
    pos_avg = np.mean(value_matrix[-n:,0])
    neg_avg = np.mean(value_matrix[:n,0])
    pos_corr_matrix = np.corrcoef(value_matrix[-n:], rowvar = False)
    mid_corr_matrix = np.corrcoef(value_matrix[n:-n], rowvar = False)
    neg_corr_matrix = np.corrcoef(value_matrix[:n], rowvar = False)
    pos_score = abs(pos_corr_matrix[0,1:]) - abs(neg_corr_matrix[0,1:]) - abs(mid_corr_matrix[0,1:])
    neg_score = abs(neg_corr_matrix[0,1:]) - abs(pos_corr_matrix[0,1:]) - abs(mid_corr_matrix[0,1:])
    print(len(pos_score))
    for i in range(0, len(pos_score)):
        if pos_score[i] == np.max(pos_score) and G.has_edge(output_node, i) == False: #we could gt a threshold value here, to not introduce wastefull nodes
            weight = pos_score[i] * pos_avg * np.sign(pos_corr_matrix[0,i+1])
            G.add_edge(output_node, i, {'weight':weight})
            print("Added node for positive correlation between positive residual and node "+ str(i))
        if neg_score[i] == np.max(neg_score) and G.has_edge(output_node, i) == False:
            weight = neg_score[i] * neg_avg * np.sign(neg_corr_matrix[0,i+1])
            G.add_edge(output_node, i, {'weight':weight})
            print("Added node for positive correlation between negative residual and node "+ str(i))
    return

def residual_patching(new_nodes, residual_iterations, residuals, output_node, input_nodes, L1, L2, lr, hidden_nodes, X_train):
    number = np.max(hidden_nodes) + 1 #gets the last node added
    connections = 1 #we will not use these...
    steps_back = 1  #we will not use there...
    nodes = np.zeros(new_nodes)
    for n in range(0, new_nodes):
        node = int(number + n)
        nodes[n] = node
        _ = generate_node(node, connections, steps_back, input_nodes, output = False, residual= True)
        if activation_type == "Sigmoid":
            weight = np.random.normal(0, scale=0.001)
        else:
            connected_nodes = np.asarray(G.neighbors(output_node))
            r = np.sqrt(2/len(connected_nodes))
            weight = np.random.uniform(-r,r)
        G.add_edge(output_node, node, {'weight':weight})
    value_matrix, predictions, total_errors = iterate(residual_iterations, X_train, residuals, nodes, L1, L2, lr, output_node, residual = True)
    hidden_nodes = np.concatenate([hidden_nodes, nodes])
    return hidden_nodes
