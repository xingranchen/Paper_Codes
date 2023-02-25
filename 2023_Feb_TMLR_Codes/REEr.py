import numpy as np
import random
import itertools
import copy
import math
import networkx as nx


class Node(object):
    def __init__(self, in_Q0, in_Q, in_Y, in_Inform, in_State_previous, in_State, in_u, in_w,
                 in_e, in_v, in_v_p, in_Neighbor_previous, in_Neighbor, in_Neighbor_alpha):
        self.Q0 = in_Q0  # the indicator of quarantine before tests
        self.Q = in_Q  # the indicator of quarantine after tests
        self.Y = in_Y  # observations in the current day, Y = -1, 0, 1
        self.Inform = in_Inform  # node i should be tested if he is informed
        self.State_previous = in_State_previous  # the true state of nodes in the previous day
        self.State = in_State  # the true state of nodes on the current day 
        self.u = in_u  # the prior probability
        self.w = in_w  # the posterior probability
        self.e = in_e  # the updated posterior probability
        self.v = in_v  # the true probability
        self.v_p = in_v_p  # the copy of in_v
        self.Neighbor_previous = in_Neighbor_previous  # the neighbors on the current day before tests
        self.Neighbor = in_Neighbor  # the neighbors on the current day after tests
        self.Neighbor_alpha = in_Neighbor_alpha  # the sets of neighbors by alpha-linking backward updating (see Appendix K in the paper)



def find_neighbors(i, node_list, network, n, keyword):
    # This function is to find neighbors of each node.
    neighbor = []
    local_neighbor = neighbor.append
    for each in range(n):
        if network[i][each] == 1:
            if keyword == 'previous':  # find the neighbors in the previous day
                if node_list[each].Q0 == 0:
                    local_neighbor(each)
            elif keyword == 'current':  # find the neighbors on the current day
                if node_list[each].Q == 0:
                    local_neighbor(each)
    return neighbor

# r = lambda: 1/r is the mean duration in the infectious state 
# r_a = gamma: 1/r_a is the mean duration in the latent state
def state_change(i, Node_list, beta_a, r_a, r):
    # This function is to describe how the states change.
    # The states follow the order [I_a, L, RD, S]
    # I_a: infectious but asymptomatic
    # L: latency state
    # RD: recover or death
    # S: susceptible
    
    sigma_i = 'S' 
    
    if Node_list[i].State == 'S':
        if len(Node_list[i].Neighbor_previous) > 0: 
            temp = 1
            for each in Node_list[i].Neighbor_previous:
                p = 0  # p is the probability that being infected by node "each"
                if Node_list[each].State == 'I_a':  
                    p = beta_a # If the node is in state I_a, we set p = beta, beta is the transmission rate.
                temp = temp * (1 - p)
            q = 1 - temp  # the probability that the node is infected by its neighbors
            if random.uniform(0, 1) > q:  # if node i is not infected
                sigma_i = 'S'
            else:
                sigma_i = 'L'  # the node is infected
    elif Node_list[i].State == 'L':  # if the node is in L state
        if random.uniform(0, 1) <= r:  # if the node becomes infectious
            sigma_i = 'I_a'
        else:
            sigma_i = 'L'
    elif Node_list[i].State == 'I_a':
        if random.uniform(0, 1) <= r_a:  # if the node recovers or dies
            sigma_i = 'RD'
        else:
            sigma_i = 'I_a'
    elif Node_list[i].State == 'RD':
        sigma_i = 'RD'  # the recovered node will not be infected again
    return sigma_i


# forward updating rule
def compute_prob(i, Node_list, beta_a, r, r_a, Step):
    P = 4 * [0]
    if Step == 'forward':  # compute u = w * P
        xi = 0
        if len(Node_list[i].Neighbor) > 0:  # consider the neighbors on the current day (after tests)
            xi0 = 1
            for each in Node_list[i].Neighbor:
                xi0 = xi0 * (1 - Node_list[each].w[0] * beta_a)
            xi = 1 - xi0
        P[3] = Node_list[i].w[3] * (1 - xi)
        P[1] = Node_list[i].w[1] * (1 - r) + Node_list[i].w[3] * xi
        P[0] = Node_list[i].w[0] * (1 - r_a) + Node_list[i].w[1] * r
        P[2] = Node_list[i].w[2] + Node_list[i].w[0] * r_a
    elif Step == 'backward':  # compute w = e * P
        if Node_list[i].Y == -1:
            xi = 0
            if len(Node_list[i].Neighbor_previous) > 0:  # consider the neighbors in the previous day
                xi0 = 1
                for each in Node_list[i].Neighbor_previous:
                    xi0 = xi0 * (1 - Node_list[each].e[0] * beta_a)
                xi = 1 - xi0
            P[3] = Node_list[i].e[3] * (1 - xi)
            P[1] = Node_list[i].e[1] * (1 - r) + Node_list[i].e[3] * xi
            P[0] = Node_list[i].e[0] * (1 - r_a) + Node_list[i].e[1] * r
            P[2] = Node_list[i].e[2] + Node_list[i].e[0] * r_a
        elif Node_list[i].Y == 0:
            xi = 0
            if len(Node_list[i].Neighbor_previous) > 0:
                xi0 = 1
                for each in Node_list[i].Neighbor_previous:
                    xi0 = xi0 * (1 - Node_list[each].e[0] * beta_a)
                xi = 1 - xi0
            P[3] = Node_list[i].e[3] * (1 - xi)
            P[1] = Node_list[i].e[1] + Node_list[i].e[3] * xi
            P[0] = 0
            P[2] = Node_list[i].e[2] + Node_list[i].e[0]
        elif Node_list[i].Y == 1:
            xi = 0
            if len(Node_list[i].Neighbor_previous) > 0:
                xi0 = 1
                for each in Node_list[i].Neighbor_previous:
                    xi0 = xi0 * (1 - Node_list[each].e[0] * beta_a)
                xi = 1 - xi0
            P[3] = Node_list[i].e[3] * (1 - xi)
            P[1] = Node_list[i].e[3] * xi
            P[0] = Node_list[i].e[0] + Node_list[i].e[1]
            P[2] = Node_list[i].e[2]
    elif Step == 'known':  # the probability vector is used to calculate $\rho$ defined in (42)
        xi = 0
        if len(Node_list[i].Neighbor_previous) > 0:
            xi0 = 1
            for each in Node_list[i].Neighbor_previous:
                xi0 = xi0 * (1 - Node_list[each].w[0] * beta_a)
                xi = 1 - xi0
        P[3] = Node_list[i].w[3] * (1 - xi)
        P[1] = Node_list[i].w[1] * (1 - r) + Node_list[i].w[3] * xi
        P[0] = Node_list[i].w[0] * (1 - r_a) + Node_list[i].w[1] * r
        P[2] = Node_list[i].w[2] + Node_list[i].w[0] * r_a
    elif Step == 'True':  # to calculate the true probability
        xi = 0
        if len(Node_list[i].Neighbor) > 0:
            xi0 = 1
            for each in Node_list[i].Neighbor:
                xi0 = xi0 * (1 - Node_list[each].v_p[0] * beta_a)
                xi = 1 - xi0
        P[3] = Node_list[i].v_p[3] * (1 - xi)
        P[1] = Node_list[i].v_p[1] * (1 - r) + Node_list[i].v_p[3] * xi
        P[0] = Node_list[i].v_p[0] * (1 - r_a) + Node_list[i].v_p[1] * r
        P[2] = Node_list[i].v_p[2] + Node_list[i].v_p[0] * r_a
    return P


def calculate_rho(i, Psi, Lambda, Psi_state, sigma_i, Node_list, beta_a, r, r_a):
    Node_list_copy = copy.deepcopy(Node_list)
    Node_list_copy[i].w = np.zeros(4)
    Node_list_copy[i].w[sigma_i] = 1
    if sigma_i == 0:  # sigma_i is the state in the previous day
        Node_list_copy[i].State = 'I_a'
    elif sigma_i == 1:
        Node_list_copy[i].State = 'L'
    elif sigma_i == 2:
        Node_list_copy[i].State = 'RD'
    elif sigma_i == 3:
        Node_list_copy[i].State = 'S'

    indicator = 0
    if i in Psi:
        Psi.remove(i)
        indicator = 1

    permutation2 = itertools.product('0123', repeat=len(Psi))  
    permutation3 = itertools.product('AB', repeat=len(Lambda))
    if indicator == 1:
        Psi.append(i)

    sum_rho = 0
    for per in permutation2:
        rho_mid = 1
        psi_state = list(map(int, list(per)))
        if indicator == 1:
            psi_state.append(sigma_i) 
        kk = 0
        for count in psi_state:  
            Node_list_copy[Psi[kk]].w = np.zeros(4)
            Node_list_copy[Psi[kk]].w[count] = 1
            rho_mid = rho_mid * Node_list[Psi[kk]].w[count] 
            kk = kk + 1
        for each in permutation3:
            rho1 = 1
            Node_list_copy1 = copy.deepcopy(Node_list_copy)
            kk = 0
            for every in each:
                Node_list_copy1[Lambda[kk]].w = np.zeros(4)
                prob_of_state = 1
                if every == 'A':
                    prob_of_state = Node_list[Lambda[kk]].w[0]
                    Node_list_copy1[Lambda[kk]].w[0] = 1
                elif every == 'B':
                    prob_of_state = Node_list[Lambda[kk]].w[1] + Node_list[Lambda[kk]].w[2] \
                                    + Node_list[Lambda[kk]].w[3]
                    Node_list_copy1[Lambda[kk]].w[3] = 1
                rho1 = rho1 * prob_of_state
                kk = kk + 1
            kk = 0
            for each in Psi:
                vec = compute_prob(each, Node_list_copy1, beta_a, r, r_a, 'known')
                rho1 = rho1 * vec[Psi_state[kk]]
                kk = kk + 1
            sum_rho = sum_rho + rho_mid * rho1
    if sum_rho > 10 ** 10:  # this condition can not be reached
        sum_rho = 0 
    return sum_rho


def delta(i, number1, Node_list):
    # delta is defined in Appendix G
    delta0 = 0
    if Node_list[i].Y == 0:
        if number1 in [1, 2, 3]:
            delta0 = 1
    elif Node_list[i].Y == 1:
        if number1 == 0:
            delta0 = 1
    return delta0



####################################
# In the rest of codes, we consider Watts-Strogatz networks. One can generate the code to other networks such as scale free networks and stochastic block model.

N = 300  # the total number of nodes
iteration = 1
Iteration = 2000  
Time_horizon = 150  
beta_s = 0.4  
gamma_a = 1
beta_a = beta_s * gamma_a
r = 1 / 1
r_a = 1 / 7
Infection_State = ['I_a']
Infection_State_count = ['I_a', 'L']
Space = ['I_a', 'L', 'RD', 'S']
Initial_Infected_nodes = 3  # the number of initial nodes
Budget = 0  # the initial budget 
alpha = 0.8  # the parameter for alpha-linking backward updating (see Appendix K in the paper)
Index = 2  # Index + 1 = ell (the unregulated delay, defined in Section 5.1)

G = nx.watts_strogatz_graph(N, 4, 0)
Network = np.array(nx.adjacency_matrix(G).todense())

Number_of_Kits = 0
Num_Daily_Tests = np.zeros((1, Time_horizon))
Num_Cumulative = np.zeros((1, Time_horizon))
Num_Cumulative[0][0] = Initial_Infected_nodes * Iteration
Num_Daily_Finds = np.zeros((1, Time_horizon))
Estimation = np.zeros((1, Time_horizon))

while iteration <= Iteration:
    Node_list = []
    for i in range(N):  # initial values for each node
        Node_list.append(Node(0, 0, -1, 0, 'S', 'S', [Initial_Infected_nodes / N, 0, 0, 1 - Initial_Infected_nodes / N],
                              [Initial_Infected_nodes / N, 0, 0, 1 - Initial_Infected_nodes / N],
                              [Initial_Infected_nodes / N, 0, 0, 1 - Initial_Infected_nodes / N], [0, 0, 0, 1],
                              [0, 0, 0, 1], [], [], []))

    Initial = random.sample(range(0, N), Initial_Infected_nodes)  # we choose initial infectious nodes randomly
    for i in Initial:
        Node_list[i].State = 'I_a'
        Node_list[i].State_previous = 'I_a'
        Node_list[i].v = [1, 0, 0, 0]
        Node_list[i].v_p = [1, 0, 0, 0]

    Network_alpha = np.zeros((N, N))  # create a network under the alpha-linking backward updating
    for each in range(N):
        for item in range(N):
            if each < item:
                if Network[each][item] == 1:
                    Network_alpha[each][item] = (random.uniform(0, 1) <= alpha)
                    Network_alpha[item][each] = Network_alpha[each][item]

    for each in range(N):  
        Node_list[each].Neighbor_previous = find_neighbors(each, Node_list, Network, N, 'previous')
        Node_list[each].Neighbor = find_neighbors(each, Node_list, Network, N, 'current')
        Node_list[each].Neighbor_alpha = find_neighbors(each, Node_list, Network_alpha, N, 'previous')

    k = 0
    num_kits = 0
    Find_positive = list(Initial)  # cumulative infections
    Num_positive_time = np.zeros((1, Time_horizon))
    Num_positive_time[0][0] = len(Initial)

    while k <= Time_horizon - 1:
        # in the beginning of each day, states of nodes change
        A = N * ['S']
        for each in range(N):
            a = state_change(each, Node_list, beta_a, r_a, r)
            A[each] = a

        for each in range(N):
            Node_list[each].State = A[each]
            Node_list[each].Y = -1

        if k == Index+1:  # recall that ell = Index + 1
            Initial_mid = []
            for i in range(N):
                if Node_list[i].State == 'I_a':
                    Initial_mid.append(i)  # find all infectious nodes
            Initial_choose = []
            if len(Initial_mid) >= 1:
                Initial_choose1 = random.sample(range(0, len(Initial_mid)), 1)
                for item in Initial_choose1:
                    Initial_choose.append(Initial_mid[item])  # we assume that one infectious node is known to the algorithm
            else:
                Initial_choose = random.sample(range(0, N), 1)  
            for i in range(N):
                if i == Initial_choose[0]:
                    Node_list[i].e = [1, 0, 0, 0]
                    Node_list[i].w = [1, 0, 0, 0]
                    Node_list[i].u = [1, 0, 0, 0]
                    Node_list[i].Inform = 1
                else:
                    Node_list[i].e = [0, 0, 0, 1]
                    Node_list[i].w = [0, 0, 0, 1]
                    Node_list[i].u = [0, 0, 0, 1]

        for each in range(N):
            Node_list[each].Q = Node_list[each].Q0

        for each in range(N):
            if Node_list[each].Inform == 1:
                if Node_list[each].State in Infection_State:  # if the node is tested positive
                    Node_list[each].Q = 1  # the node is quarantined immediately
                    Node_list[each].Y = 1  # the observation of the positive node becomes to 1
                else:  # if node i is tested negative
                    Node_list[each].Q = 0
                    Node_list[each].Y = 0
        
        # now, we calculate the updated posterior probabilities
        O_t = []  # the set of nodes who has new observations
        for count in range(N):  
            if Node_list[count].Y > -1:
                if Node_list[count].Q0 == 0:
                    O_t.append(count)
        # the neighbors of nodes in O_t, i.e., the nodes who are affected by nodes in O_t
        Affect_by_O_t = O_t
        for each in O_t:
            Affect_by_O_t = Affect_by_O_t + Node_list[each].Neighbor_previous
        Affect_by_O_t = list(set(Affect_by_O_t))

        for i in Affect_by_O_t:  
            Node_list[i].Neighbor_alpha.append(i)
            Psi = list(set(O_t) & set(Node_list[i].Neighbor_alpha))
            Pr_i_x = 4 * [0]
            Phi = []
            if len(Psi) > 0:
                for each in Psi:
                    Phi.append(each)  # $\Phi$ is the set of neighbors of neighbors
                    Phi.extend(Node_list[each].Neighbor_alpha)  
                Phi = list(set(Phi)) 
            if i in Phi:  # Note that $\Phi$ can not contain i.
                Phi.remove(i)
            Lambda = []
            if len(Phi) > 0:
                Lambda = list(set(Phi) - set(Psi))
            if len(Psi) > 0:
                permutation1 = itertools.product('0123', repeat=len(Psi)) 
                for per in permutation1:
                    Psi_state = list(map(int, list(per)))
                    delta_function = 1
                    kk = 0
                    for each in Psi:
                        delta_function = delta_function * delta(each, Psi_state[kk], Node_list)
                        kk = kk + 1
                    for state_i in range(4):
                        Pr_i_x[state_i] = Pr_i_x[state_i] + delta_function * calculate_rho(i, Psi, Lambda, Psi_state,
                                                                                           state_i, Node_list, beta_a,
                                                                                           r, r_a)
            Pr_i = 0 
            for count in range(4):
                Pr_i = Pr_i + Pr_i_x[count] * Node_list[i].w[count]
            if Pr_i > 0:
                for count in range(4):
                    Node_list[i].e[count] = Pr_i_x[count] * Node_list[i].w[count] / Pr_i
            else:
                Node_list[i].e = Node_list[i].w

        for i in range(N):
            if i not in Affect_by_O_t:
                Node_list[i].e = Node_list[i].w

        for i in range(N):  
            if sum(Node_list[i].e) > 1.001:  # the condition can not be reached
                print(Node_list[i].e, iteration, k, i, 'e')
            Node_list[i].w = compute_prob(i, Node_list, beta_a, r, r_a, 'backward')
            if sum(Node_list[i].w) > 1.001:  # the condition can not be reached
                print(Node_list[i].w, iteration, k, i, 'w')
            Node_list[i].Neighbor = find_neighbors(i, Node_list, Network, N, 'current')

        num_estimation = 0
        sum_estimation = 0
        for i in range(N):
            Node_list[i].u = compute_prob(i, Node_list, beta_a, r, r_a, 'forward')
            Node_list[i].v = compute_prob(i, Node_list, beta_a, r, r_a, 'True')
            if sum(Node_list[i].u) > 1.001:  # the condition can not be reached
                print(Node_list[i].u, iteration, k, i, 'u')
            if sum(Node_list[i].v) > 1.001:  # the condition can not be reached
                print(Node_list[i].v, iteration, k, i, 'v')

            if Node_list[i].Q == 0:
                num_estimation = num_estimation + 1
                sum_estimation = sum_estimation + (Node_list[i].u[0] - Node_list[i].v[0]) ** 2 + (
                        Node_list[i].u[1] - Node_list[i].v[1]) ** 2 + (Node_list[i].u[2] - Node_list[i].v[2]) ** 2 + (
                                         Node_list[i].u[3] - Node_list[i].v[3]) ** 2
        Estimation[0][k] = Estimation[0][k] + sum_estimation / num_estimation

        Budget = 0
        if k <= Index:  # let the disease run for $\ell$ days
            Budget = 0
        else:
            for i in range(N):
                if Node_list[i].Q == 0:
                    Budget = Budget + Node_list[i].v[0]  # the budget equals to the number of expected infectious nodes
        Sum_u = 0
        score = N * [0]  # calculate the reward
        for i in range(N):
            if Node_list[i].Q == 0:
                score_1 = Node_list[i].u[0] * (1 - r_a)
                score_2 = 0
                if len(Node_list[i].Neighbor):
                    for neighbor in Node_list[i].Neighbor:
                        dummy = 1
                        if len(Node_list[neighbor].Neighbor) > 1:
                            Node_list[neighbor].Neighbor.remove(i)
                            for neighbor_neighbor in Node_list[neighbor].Neighbor:
                                dummy = dummy * (1 - beta_s * Node_list[neighbor_neighbor].u[0])
                            Node_list[neighbor].Neighbor.append(i)
                        score_2 = score_2 + Node_list[neighbor].u[3] * beta_s * Node_list[i].u[0] * dummy
                score[i] = score_1 + score_2
            Sum_u = Sum_u + score[i]

        K_prime = 0
        if Sum_u > 0:
            for i in range(N):
                Node_list[i].Inform = 0
                if Node_list[i].Q == 0:
                    Node_list[i].Inform = (random.uniform(0, 1) <= Budget * (score[i]/Sum_u))
                    K_prime = K_prime + max(0, Budget * (score[i]/Sum_u) - 1)
        else:
            for i in range(N):
                Node_list[i].Inform = 0
            K_prime = Budget

        K_prime_int = 0

        if K_prime > 0:
            K_prime_int = math.floor(K_prime)
            q = K_prime - K_prime_int
            if random.uniform(0, 1) <= q:
                K_prime_int = K_prime_int + 1
            Ready = []
            for each in range(N):
                if Node_list[each].Q == 0:
                    if Node_list[each].Inform == 0:
                        Ready.append(each)
            if len(Ready) >= K_prime_int:
                Ready1 = random.sample(Ready, K_prime_int)
            else:
                Ready1 = Ready
            for each in Ready1:
                Node_list[each].Inform = 1

        for each in range(N):
            if Node_list[each].State in Infection_State_count:  # count infectious nodes
                if each not in Find_positive:  # the node has not been found
                    Find_positive.append(each)
        k = k + 1

        for each in range(N):
            Node_list[each].Q0 = Node_list[each].Q
            Node_list[each].Neighbor_previous = []
            Node_list[each].Neighbor_previous = Node_list[each].Neighbor
            Node_list[each].Neighbor_alpha = find_neighbors(each, Node_list, Network_alpha, N, 'previous')
            Node_list[each].w = Node_list[each].u
            Node_list[each].v_p = Node_list[each].v

        Network_alpha = np.zeros((N, N))
        for each in range(N):
            for item in range(N):
                if each < item:
                    if Network[each][item] == 1:
                        Network_alpha[each][item] = (random.uniform(0, 1) <= alpha)
                        Network_alpha[item][each] = Network_alpha[each][item]

        if k < Time_horizon:
            Num_positive_time[0][k] = len(Find_positive)
            Num_Cumulative[0][k] = Num_Cumulative[0][k] + Num_positive_time[0][k]

    Number_of_Kits = Number_of_Kits + num_kits
    print(iteration)
    iteration = iteration + 1

Num_Cumulative = Num_Cumulative / Iteration
Number_of_Kits = Number_of_Kits / Iteration
Estimation = Estimation / Iteration
print('Number of Tests: ', Number_of_Kits)
print('Cumulative Infections: ', Num_Cumulative)
print('Estimation: ', Estimation)