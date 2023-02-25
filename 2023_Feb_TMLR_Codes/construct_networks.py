import numpy as np
import networkx as nx
import random

#######################
# parameters
N = 300  # number of nodes
degree_WS = 4
delta_WS = 0.03
alpha_SF = 2.3
K_SBM = 10
p1_SBM = 0.05
p2_SBM = 0.01
K_VSBM = 10
p1_VSBM = 0.05
p2_VSBM = 0.01

#######################
# Watts-Strogatz Networks
G1 = nx.watts_strogatz_graph(N, degree_WS, delta_WS)
Network1 = np.array(nx.adjacency_matrix(G1).todense())

#######################
# Scale-free Networks
s = nx.utils.powerlaw_sequence(N, alpha_SF) 
G2 = nx.expected_degree_graph(s, selfloops=False)
Network2 = np.array(nx.adjacency_matrix(G2).todense())

#######################
# Stochastic Block Model
Network3 = np.zeros(shape=(N, N))
Index = N*[0]
for each in range(N):
    Index[each] = each % K_SBM
for each in range(N):
    for item in range(N):
        if Index[each] == Index[item]:
            Network3[each][item] = (random.uniform(0, 1) < p1_SBM)
            Network3[item][each] = Network3[each][item]
        else:
            Network3[each][item] = (random.uniform(0, 1) < p2_SBM)
            Network3[item][each] = Network3[each][item]
for each in range(N):
    Network3[each][each] = 0

#######################
# A Variant of Stochastic Block Model
Network4 = np.zeros(shape=(N, N))
Index = N*[0]
for each in range(N):
    Index[each] = each % K_VSBM

for each in range(N):
    for item in range(N):
        if Index[each] == Index[item]:
            Network4[each][item] = (random.uniform(0, 1) < p1_VSBM)
            Network4[item][each] = Network4[each][item]
        elif (Index[each] == K_VSBM-1) and (Index[item] == 0):
            Network4[each][item] = (random.uniform(0, 1) < p2_VSBM)
            Network4[item][each] = Network4[each][item]
        elif Index[each] == Index[item] - 1:
            Network4[each][item] = (random.uniform(0, 1) < p2_VSBM)
            Network4[item][each] = Network4[each][item]
for each in range(N):
    Network4[each][each] = 0





