import networkx as nx
import numpy as np
import random

def generate_node(last_number, connections, steps_back):
    G.add_node(last_number)
    G.node[last_number]['bias'] = 0
    if last_number <= connections:
        connected_nodes = random.sample(range(0, last_number), last_number)
    else:
        connected_nodes = random.sample(range(max(0,last_number - steps_back),last_number), connections)
    if activation_type == "Sigmoid":
        weight = np.random.normal(0, scale=0.001)
    else:
        r = np.sqrt(2/len(connected_nodes))
        weight = np.random.uniform(-r,r)
    for i in connected_nodes:
        G.add_edge(last_number,i, {'weight':weight})
    last_number += 1
    return last_number

def generate_input_nodes(inputs):
    for i in range(0,inputs):
        G.add_node(i)
    return

def make_graph(input_nodes, hidden_nodes, hidden_connections,steps_back, output_connections, output_node):
    generate_input_nodes(input_nodes)
    last_number = input_nodes
    for i in range(0, hidden_nodes):
        last_number = generate_node(last_number, hidden_connections, steps_back)
    last_number = generate_node(output, output_connections, output_connections)
    hidden_nodes = np.arange(hidden_nodes) + input_nodes
    return hidden_nodes, G
