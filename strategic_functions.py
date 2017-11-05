import numpy as np
import networkx

def new_edge(value_matrix, hidden_nodes, n):
    value_matrix = value_matrix[np.argsort(value_matrix[:,0])]
    pos_avg = np.mean(value_matrix[-n:,0])
    neg_avg = np.mean(value_matrix[:n,0])
    pos_corr_matrix = np.corrcoef(value_matrix[-n:], rowvar = False)
    mid_corr_matrix = np.corrcoef(value_matrix[n:-n], rowvar = False)
    neg_corr_matrix = np.corrcoef(value_matrix[:n], rowvar = False)
    pos_score = abs(pos_corr_matrix[0,1:]) - abs(neg_corr_matrix[0,1:]) - abs(mid_corr_matrix[0,1:])
    neg_score = abs(neg_corr_matrix[0,1:]) - abs(pos_corr_matrix[0,1:]) - abs(mid_corr_matrix[0,1:])
    output = np.max(hidden_nodes) + 1
    for i in range(0, len(pos_score)):
        if pos_score[i] == np.max(pos_score) and G.has_edge(output, hidden_nodes[i]) == False: #we could gt a threshold value here, to not introduce wastefull nodes
            weight = pos_score[i] * pos_avg * np.sign(pos_corr_matrix[0,i+1])
            G.add_edge(output, hidden_nodes[i], {'weight':weight})
            print("Added node for positive correlation between positive residual and node "+ str(hidden_nodes[i]))
        if neg_score[i] == np.max(neg_score) and G.has_edge(output, hidden_nodes[i]) == False:
            weight = neg_score[i] * neg_avg * np.sign(neg_corr_matrix[0,i+1])
            G.add_edge(output, hidden_nodes[i], {'weight':weight})
            print("Added node for positive correlation between negative residual and node "+ str(hidden_nodes[i]))
