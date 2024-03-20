import numpy as np

def sample_transition(mdp, x, a):
    X = mdp[0]
    P = mdp[2]
    C = mdp[3]
    x_line = np.random.choice(len(X), p=P[a][x, :])
    return (x, a, C[x, a], x_line)

def egreedy(Q, eps=0.1):
    if np.random.rand() <= eps:
        return np.random.choice(len(Q))
    else:
        policy = np.isclose(Q, np.min(Q)).astype(int)
        probs = policy / np.sum(policy)
        return np.random.choice(np.arange(len(Q)), p=probs)
    
def mb_learning(mdp, n, qinit, pinit, cinit):
    X = mdp[0]
    A = mdp[1]
    gamma = mdp[4]
    
    Q = qinit
    P = pinit
    C = cinit

    x = np.random.choice(len(X))

    N = np.zeros((len(X), len(A)))

    for _ in range(n):
        a = egreedy(Q[x, :], eps=0.15)
        (_, _, cost, next_x) = sample_transition(mdp, x, a)

        N[x, a] += 1
        alpha = 1 / (N[x, a] + 1)

        C[x, a] = C[x, a] + alpha * (cost - C[x, a])

        II = np.zeros(len(X))
        II[next_x] = 1

        P[a][x, :] = P[a][x, :] + alpha * (II - P[a][x, :])

        Q[x, a] = C[x, a] + gamma * (P[a][x, :] @ np.min(Q, axis=1, keepdims=True))

        x = next_x

    return (Q, P, C)

def qlearning(mdp, n, qinit):
    Q = qinit
    alpha = 0.3
    gamma = mdp[4]
    x = np.random.choice(len(mdp[0]))
    for _ in range(n):
        a = egreedy(Q[x, :], eps=0.15)
        (_, _, c, x_line) = sample_transition(mdp, x, a)
        Q[x, a] += alpha * (c + gamma * np.amin(Q[x_line, :]) - Q[x, a])
        x = x_line
    return Q

def sarsa(mdp, n, qinit):
    Q = qinit
    alpha = 0.3
    gamma = mdp[4]

    x = np.random.choice(len(mdp[0]))
    a = egreedy(Q[x, :], eps=0.15)

    for _ in range(n):
        (_, _, c, next_x) = sample_transition(mdp, x, a)
        next_a = egreedy(Q[next_x, :], eps=0.15)
        
        Q[x, a] += alpha * (c + gamma * Q[next_x, next_a] - Q[x, a])
        
        x = next_x
        a = next_a

    return Q

# We can see that all algorithm plots reach an error close to 0, and thus learn the optimal policy.
# We also see that the model-based algorithm has the lowest error in the end. It updates the model directly, learning the probabilities and costs of the MDP. This leads to a more accurate Q-function, since it's the same method that is used to generate the optimal Q-function trough Value Iteration.
# We can also verify that Q-learning and SARSA have similar performance in the beginning. This is expected, as both are temporal diference algorithms.
# Also we can see that SARSA tends to increase in the end. This may happen since it's an on-policy algorithm, and the policy may not be optimal. This is not the case for Q-learning, which is an off-policy algorithm.


