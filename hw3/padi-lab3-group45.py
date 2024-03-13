
import numpy as np

def load_pomdp(filename, gamma):
    data = np.load(filename)
    X = tuple(data['X'])
    A = tuple(data['A'])
    Z = tuple(data['Z'])
    P = tuple(data['P'])
    O = tuple(data['O'])
    return (X, A, Z, P, O, data['c'] , gamma)

def gen_trajectory(pomdp, x0, n):
    X = pomdp[0]
    A = pomdp[1]
    Z = pomdp[2]
    P = pomdp[3]
    O = pomdp[4]
    c = pomdp[5]
    gamma = pomdp[6]

    states = np.zeros((n + 1), dtype=int)
    actions = np.zeros((n), dtype=int)
    observations = np.zeros((n), dtype=int)

    states[0] = x0

    for i in range(n):
        actions[i] = int(np.random.choice(len(A)))
        states[i+1] = int(np.random.choice(len(X), p=P[int(actions[i])][int(states[i])]))
        observations[i] = int(np.random.choice(len(Z), p=O[int(actions[i])][int(states[i+1])]))

    return (states, actions, observations)

import numpy.random as rand

def belief_update(pomdp, belief, action, observation):
    Pa = pomdp[3][action]
    Oa = pomdp[4][action]
    b = belief @ Pa @ np.diag(Oa[:, observation])
    return b / np.sum(b)

def repeated_belief(beliefs, new_belief):
    for j in range(len(beliefs)):
        if (beliefs[j] == new_belief).all() or np.linalg.norm(new_belief - beliefs[j]) < 10e-3:
            return False
    return True

def sample_beliefs(pomdp, n):
    random_state = np.random.choice(len(pomdp[0]))
    trajectory = gen_trajectory(pomdp, random_state, n)
    belief = np.ones((1, len(pomdp[0]))) / len(pomdp[0])
    beliefs = (belief,)
    for i in range(n):
        belief = belief_update(pomdp, belief, trajectory[1][i], trajectory[2][i])
        if repeated_belief(beliefs, belief):
            beliefs += (belief,)
    return beliefs

def value_iteration(mdp):
    X = mdp[0]
    A = mdp[1]
    P = mdp[3]
    c = mdp[5]
    gamma = mdp[6]

    J = np.zeros((len(X), 1))
    cur_err = 1.0
    stop_err = 10e-8

    while cur_err > stop_err:
        #Auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))

        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)

        #Compute minimum column wise
        Jnew = np.min(Q, axis=1, keepdims=True)

        #Compute error
        cur_err = np.linalg.norm(J - Jnew)

        J = Jnew
    
    return J

def solve_mdp(pomdp):
    X = pomdp[0]
    A = pomdp[1]
    P = pomdp[3]
    c = pomdp[5]
    gamma = pomdp[6]

    J_star = value_iteration(pomdp)
    aux = np.zeros((len(X), len(A)))
    for a in range(len(A)):
        for x in range(len(X)):
            aux[x, a] = P[a][x] @ J_star
        
    Q_star = c + gamma * aux
    return Q_star

def get_heuristic_action(belief, solution, heuristic):
    if (heuristic == "mls"):
        q_optimal = Q[np.argmax(belief), :]
        return np.random.choice(np.argwhere(q_optimal == np.amin(q_optimal)).flatten())
    elif (heuristic == "av"):
        votes = np.zeros(solution.shape[1])
        pi = np.argmin(solution, axis = 1)
        for a in range(solution.shape[1]):
            votes[a] = b[:, pi == a].sum()
        return np.random.choice(np.argwhere(votes == np.amax(votes)).flatten())
    elif (heuristic == "q-mdp"):
        aux = (b @ solution).flatten()
        return np.random.choice(np.argwhere(aux == np.amin(aux)).flatten())
    else:
        return ValueError("Heuristic not found")

def solve_fib(pomdp):
    X = pomdp[0]
    A = pomdp[1]
    Z = pomdp[2]
    P = pomdp[3]
    O = pomdp[4]
    c = pomdp[5]
    gamma = pomdp[6]

    Q_prev_fib = np.zeros((len(X), len(A)))
    Q_fib = np.zeros((len(X), len(A)))
    Q_fib_aux = np.zeros((len(X), len(A)))
    while True:
        for a in range(len(A)):
            for x in range(len(X)):
                aux_z = np.zeros((len(Z)))
                for z in range(len(Z)):
                    aux_min = np.zeros((len(X), len(A)))
                    for al in range(len(A)):
                        for xl in range(len(X)):
                            aux_min[x, a] += P[a][x, xl] * O[a][xl, z] * Q_fib_aux[xl, al]
                    
                    #somar os x
                    aux_min_sum = np.sum(aux_min, axis=0)
                    #ver o menor pela acao
                    aux_z[z] = np.min(aux_min)
                
                Q_fib_aux[x, a] = np.sum(aux_z)

        Q_prev_fib = Q_fib
        Q_fib = c + gamma * Q_fib_aux
        cur_err = np.linalg.norm(Q_prev_fib - Q_fib)
        if (cur_err < 10 ** -1):
            break

    return Q_fib

# Q-MDP ignores partial observability at the next time step. So for FIB we modify the position of the min in the Q-function.
# They are somewhat similar and tend to give the same results with differences on the border of decision. 


