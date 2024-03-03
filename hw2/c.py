import numpy as np

#Define the MDP
X = {'1P', '2P', '2NP', '3P', '3NP', '4P', '4NP'}
A = {'D'}
Pd = np.array([[0.2, 0, 0, 0.8, 0, 0, 0],
    [0, 0.2, 0, 0, 0, 0.8, 0],
    [0, 0, 0.2, 0, 0, 0, 0.8],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]])
P = (Pd)
c = np.array([[1], [1], [1], [1], [1], [1], [1]])

#X = {'0', 'A', 'B'}
#A = {'a', 'b'}
#Pa = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
#Pb = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
#P = (Pa, Pb)
#c = np.array([[1, 0.5], [0, 0], [1, 1]])

M = (X, A, P, c, 0.9)

def value_iteration(M, eps):
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]

    J = np.zeros((len(X), 1))
    err = 1.0
    n_iter = 0

    while err > eps:
        #Auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))

        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)

        #Compute minimum column wise
        Jnew = np.min(Q, axis=1, keepdims=True)

        #Compute error
        err = np.linalg.norm(J - Jnew)

        J = Jnew
        n_iter += 1
    
    print(f'Done after {n_iter} iterations')
    return np.round(J, 3)

J = value_iteration(M, 1e-8)

Inverse = np.linalg.inv(np.identity(7) - 0.9 * Pd)

print(np.round(Inverse, 3))

print(Inverse @ c)

print(J)

