import numpy as np

def load_mdp(filename, gamma):
    data = np.load(filename)
    states = ()
    for state in data["X"]:
        states += (state,)
    actions = ()
    for action in data["A"]:
        actions += (action,)
    transitions = ()
    for transition in data["P"]:
        transitions += (transition,)
    return (states, actions, transitions, data["c"], gamma)    

def noisy_policy(mdp, a, eps):
    pi = np.zeros((len(mdp[0]), len(mdp[1])))
    pi[:, a] = 1 - eps

    uni_prob = eps / (len(mdp[1]) - 1)
    pi[:, 0 : a] = uni_prob
    pi[:, a + 1 :] = uni_prob
    return pi

def evaluate_pol(MDP, pi):
    P_pi = np.zeros((len(MDP[0]), len(MDP[0])))
    c_pi = np.zeros((len(MDP[0]), 1))
    for x in range(len(MDP[0])):
        for y in range(len(MDP[0])):
            P_pi[x, y] = sum(pi[x, a] * MDP[2][a][x, y] for a in range(len(MDP[1])))
        c_pi[x, 0] = sum(pi[x, a] * MDP[3][x, a] for a in range(len(MDP[1])))
    return np.linalg.inv(np.eye(len(MDP[0])) - MDP[4] * P_pi) @ c_pi

import time

def value_iteration(mdp):
    X = mdp[0]
    A = mdp[1]
    P = mdp[2]
    c = mdp[3]
    gamma = mdp[4]

    J = np.zeros((len(X), 1))
    cur_err = 1.0
    stop_err = 10e-8

    init_time = time.time() 
    n_iter = 0

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
        n_iter += 1
    
    t = time.time() - init_time
    print(f'Execution time: {t:.3f} seconds')
    print(f'N. iterations: {n_iter}')
    
    return J

def policy_iteration(M):
    time_start = time.time()
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]
    
    pol = np.ones((len(X), len(A))) / len(A)
    quit = False
    niter = 0
    
    while not quit:
        Q = np.zeros((len(X), len(A)))
        
        cpi = np.sum(c * pol, axis=1, keepdims=True)
        Ppi = pol[:, 0, None] * P[0]
        for a in range(1, len(A)):
            Ppi += pol[:, a, None] * P[a]
            J = np.linalg.inv(np.eye(len(X)) - gamma * Ppi).dot(cpi)
        
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)
        
        Qmin = np.min(Q, axis=1, keepdims=True)
        
        pnew = np.isclose(Q, Qmin, atol=10e-8, rtol=10e-8).astype(int)
        pnew = pnew / pnew.sum(axis = 1, keepdims = True)
        
        quit = (pol == pnew).all()
        
        pol = pnew
        niter += 1
    print(f'Execution time: {time.time() - time_start:.3f} seconds')
    print(f'N. iterations: {niter}')
    return pol

import numpy.random as rand

NRUNS = 100 # Do not delete this

# Add your code here.
def simulate(mdp, pi, x0, length):
    X = mdp[0]
    A = mdp[1]
    P = mdp[2]
    c = mdp[3]
    gamma = mdp[4]

    j_x0 = 0
    for _ in range(NRUNS):
        state = x0
        for i in range(length):
            action = rand.choice(len(A), p=pi[state])
            j_x0 += (gamma ** i) * c[state, action]
            state = rand.choice(len(X), p=P[action][state])
    
    return j_x0 / NRUNS
