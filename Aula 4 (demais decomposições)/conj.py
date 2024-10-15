import numpy as np
from scipy.sparse.linalg import cg  # Conjugate Gradient solver

def conjugate_decomposition(A):
    m, n = A.shape
    k = np.linalg.matrix_rank(A)

    # Step 1: Initialize
    r = np.zeros((m, 1))
    p = np.zeros((n, 1))
    Q = np.eye(m)
    P = np.eye(n)

    for i in range(k):
        # Step 2.1: Conjugate Gradient on A^T A
        AtA = A.T @ A
        b = A.T @ r
        
        # Conjugate Gradient method
        p_i = np.zeros((n, 1))  # Initialize p_i
        p_i, _ = cg(AtA, b.flatten(), x0=p.flatten())  # Solve the CG problem
        
        q = A @ p_i
        r = r - (A.T @ q).flatten()
        alpha = np.dot(p_i.flatten(), AtA @ p_i.flatten()) / np.dot(q.flatten(), q.flatten())
        p_i = r + alpha * p_i.flatten()
        
        # Normalize p_i
        p_i = p_i / np.linalg.norm(p_i)
        
        # Step 2.2: Calculate q_i
        q_i = q / np.linalg.norm(q)
        
        # Update Q and P
        Q[:, i] = q_i.flatten()
        P[:, i] = p_i.flatten()

    # Step 3: Orthogonalize remaining vectors
    P[:, k:] = np.eye(n)[:, k:]
    Q[:, k:] = np.eye(m)[:, k:]
    
    # Step 4: Ensure orthonormality of remaining vectors
    P[:, k:] = np.linalg.qr(P[:, k:])[0]
    Q[:, k:] = np.linalg.qr(Q[:, k:])[0]
    
    # Compute Γ
    Gamma = np.zeros((m, n))
    for i in range(k):
        Gamma[i, i] = np.sqrt(np.dot(p.flatten(), AtA @ p.flatten()))

    return Q, Gamma, np.linalg.inv(P)

# Example usage
A = np.random.rand(5, 3)  # Replace with your matrix
Q, Gamma, P_inv = conjugate_decomposition(A)

print("Q:\n", Q)
print("Gamma:\n", Gamma)
print("P_inv:\n", P_inv)
