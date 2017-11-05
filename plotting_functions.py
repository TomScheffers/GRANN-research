import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

def plot(X_train, y_train, X_test, y_test, y_preds, data_size, function_type, hidden_nodes):
    nx.draw(G, with_labels = True)
    plt.show()

    X = np.linspace(-1, 1, data_size)
    if function_type == "Sinus":
        plt.plot(X, np.sin(6.2830*X), c='r',label='Underlying data')
    elif function_type == "Quadratic":
        plt.plot(X, np.power(X, 2), c='r',label='Underlying data')
    plt.plot(X_train, y_train, c='g', marker='o', ls='',label='Train data')
    plt.plot(X_test, y_preds, c='b', marker='o', ls='',label='Predictions')
    plt.ylabel('y values')
    plt.xlabel('x values')
    plt.legend()
    plt.show()
    '''
    deltas = np.zeros(len(hidden_nodes))
    for n in hidden_nodes:
        deltas[n-1] = G.node[n]['delta']
    plt.plot(hidden_nodes, deltas)
    plt.ylabel('Node deltas')
    plt.xlabel('Node number: node 0 is input, max(node) is output')
    plt.show()

    total_weights = np.zeros(len(hidden_nodes))
    for n in hidden_nodes:
        total_weights[n-1] = G.node[n]['total_weight']
    plt.plot(hidden_nodes, total_weights)
    plt.ylabel('Total absolute weight of all incoming connections')
    plt.xlabel('Node number: node 0 is input, max(node) is output')
    plt.show()'''
    return
