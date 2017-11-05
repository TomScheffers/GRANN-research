import networkx as nx
import numpy as np
import random

def generate_node(number, connections, steps_back, input_nodes, output, residual):
    if residual == True:
        connected_nodes = np.arange(input_nodes)
    elif number <= connections:
        connected_nodes = random.sample(range(0, number), number)
    elif output == False:
        connected_nodes = random.sample(range(max(0,number - steps_back),number), connections)
    elif output == True:
        connected_nodes = random.sample(G.nodes(), connections)
    G.add_node(number)
    G.node[number]['bias'] = 0
    if activation_type == "Sigmoid":
        weight = np.random.normal(0, scale=0.001)
    else:
        r = np.sqrt(2/len(connected_nodes))
        weight = np.random.uniform(-r,r)
    for i in connected_nodes:
        G.add_edge(number,i, {'weight':weight})
    number += 1
    return number

def generate_input_nodes(inputs):
    for i in range(0,inputs):
        G.add_node(i)
    return

def make_graph(output_node, input_nodes, hidden_nodes, hidden_connections,steps_back, output_connections):
    generate_input_nodes(input_nodes)
    number = input_nodes
    for i in range(0, hidden_nodes):
        number = generate_node(number, hidden_connections, steps_back, input_nodes, output = False, residual = False)
    _ = generate_node(output_node, output_connections, output_connections, input_nodes, output = True, residual = False)
    hidden_nodes = np.arange(hidden_nodes) + input_nodes
    return hidden_nodes
