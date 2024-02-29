import numpy as np
def load_chain(filename, gamma):
    data = np.array(np.load(filename), dtype=np.float64)
    states = ()
    for i in range(data.shape[0]):
        states += (str(i),)
        norm = 1 - gamma
        data[i] = data[i] * norm
        data[i] += gamma/data[i].shape[0]
    return (states, data)

def prob_trajectory(mc, trajectory):
    prob = 1
    for i in range(len(trajectory)-1):
        prob *= mc[1][int(trajectory[i]), int(trajectory[i+1])]

    return prob

import numpy as np

def stationary_dist(mc):
    eigenvalues, eigenvectors = np.linalg.eig(mc[1].T)
    for i in range(len(eigenvalues)):
        if np.isclose(eigenvalues[i], 1.0):
            return np.real(eigenvectors[:, i].T/np.sum(eigenvectors[:, i].T))

import numpy.random as rnd

def compute_dist(M_Chain, mu_0, N):
    return mu_0 @ np.linalg.matrix_power(M_Chain[1], N)

# Since every has the teleport property, it can go from every state to every state and thus is irreductible and aperiodic. Also, the sinks have links to every other state, so they can transition to every state.
# 
# If a chain is irreductible and aperiodic with stationary distribution u*, and for a certain t-> inf, lim u_0 P^t = u*,
# with u_0 being the initial distribution, then it is an ergodic chain.
# 
# The function compute_dist allows us to compute u_0 * P^t, and since we already know that all chains loaded by load_function are irreductible and aperiodic, if we verify that the result of compute_dist equals u*, we can verify if the chain is ergodic.
# 
# Since for the chains given, u* P^2000 = u*, we can prove the chains are ergodic.

import numpy.random as rnd

def simulate(mc, init_dist, N):
    initial_state = rnd.choice(mc[0], p=init_dist[0])
    trajectory = (initial_state,)
    for step in range(1, N):
        trajectory += (rnd.choice(mc[0], p=mc[1][int(trajectory[step-1])]),)

    return trajectory

import matplotlib.pyplot as plt

chain = load_chain("example.npy", 0.11)

mu_0 = np.ones(shape=(1,len(chain[0])))/len(chain[0]) # uniform distribution

path = simulate(chain, mu_0, 50000)

path =  np.array(np.sort(path), dtype=np.int64)

plt.hist(path, bins=[x for x in range(len(chain[0])+1)], density=True)
plt.plot([x + 0.5 for x in range(len(chain[0]))], stationary_dist(chain), 'ro')

plt.show()
