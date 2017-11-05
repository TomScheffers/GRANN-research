import networkx as nx
import numpy as np
from activation_functions import activation_der, activation_func
from sklearn.metrics import mean_squared_error

def set_input(x, input_nodes):
    for i in range(0,input_nodes):
        G.nodes[i]['value'] = x[i]
    return

def calculate_activation(node, activation):
    neighbors = np.asarray(G.neighbors(node))
    neighbors = [k for k in neighbors if k < node]
    forw_act = 0
    total_weight = 0
    for neig in neighbors:
        forw_act += G.node[neig]['value'] * G.edge[node][neig]['weight']
        total_weight += G.edge[node][neig]['weight']
    forw_act += G.node[node]['bias']
    if activation == True:
        forw_out = activation_func(forw_act)
        #print(node, len(neighbors), G.node[node]['bias'], forw_act, forw_out)
    else:
        forw_out = forw_act
    G.node[node]['value'] = forw_out
    G.node[node]['act'] = forw_act
    G.node[node]['total_weight'] = total_weight
    return forw_out

def forward_pass(hidden_nodes):
    for node in hidden_nodes:
        forw_out = calculate_activation(node, True)
    output = calculate_activation(np.max(hidden_nodes)+1, False)
    return output

def update_weights(node, L1, L2, lr):
    delta = G.node[node]['delta']
    G.node[node]['bias'] = G.node[node]['bias'] - lr * delta
    neighbors = np.asarray(G.neighbors(node))
    neighbors = [k for k in neighbors if k < node]
    for neig in neighbors:
        G.edge[node][neig]['weight'] = G.edge[node][neig]['weight'] - lr * delta * G.node[neig]['value'] - L1 * np.sign(G.edge[node][neig]['weight']) - L2 * G.edge[node][neig]['weight']

def calculate_delta(node):
    neighbors = np.asarray(G.neighbors(node))
    neighbors = [k for k in neighbors if k > node]
    delta = 0
    for neig in neighbors:
        delta += G.node[neig]['delta'] * G.edge[neig][node]['weight']
    G.node[node]['delta'] = delta * activation_der(G.node[node]['act'])
    return

def back_prop(error, hidden_nodes, L1, L2, lr):
    G.node[np.max(hidden_nodes)+1]['delta'] = error
    for node in hidden_nodes:
        calculate_delta(node)
    update_weights(np.max(hidden_nodes)+1, L1, L2, lr)
    for node in hidden_nodes:
        update_weights(node, L1, L2, lr)
    return

def iterate(iterations, X_train, y_train, hidden_nodes, L1, L2, lr):
    value_matrix = np.zeros((X_train.shape[0], len(hidden_nodes) + 1))
    total_errors = np.zeros(iterations)
    for iteration in range(0,iterations):
        total_error = 0
        predictions = np.zeros(X_train.shape[0])
        for i in range(0,X_train.shape[0]):
            G.node[0]['value'] = X_train[i]
            predictions[i] = forward_pass(hidden_nodes)
            error = predictions[i] - y_train[i]
            total_error += error*error
            back_prop(error, hidden_nodes[::-1], L1, L2, lr)
            if iteration == iterations - 1: #only on last iteration
                node_values = list(nx.get_node_attributes(G, 'value').values())
                value_matrix[i, 1:] = node_values[1:-1]
                value_matrix[i, 0] = error
        print(iteration, total_error/X_train.shape[0])
        total_errors[iteration] = total_error/X_train.shape[0]
    return value_matrix, predictions, total_errors

def calculate_preds(X_test, y_test, hidden_nodes):
    y_preds = np.zeros(X_test.shape[0])
    for i in range(0,X_test.shape[0]):
        G.node[0]['value'] = X_test[i]
        y_preds[i] = forward_pass(hidden_nodes)
    mse = mean_squared_error(y_test, y_preds)
    print(mse)
    return mse, y_preds
