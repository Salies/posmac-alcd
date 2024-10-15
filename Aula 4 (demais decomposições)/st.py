import numpy as np

def golub_yuan_st_decomposition(A):
    n = A.shape[0]
    
    # Initialize matrices
    L = np.eye(n)
    T = np.eye(n)
    
    # Set t11 such that t11a11 > 0, and l11 = sqrt(t11a11)
    T[0, 0] = A[0, 0]
    L[0, 0] = np.sqrt(T[0, 0] * A[0, 0])
    
    for k in range(n - 1):
        # Compute L_k and T_k based on previous computations
        Lk = L[:k+1, :k+1]
        Tk = T[:k+1, :k+1]
        
        l_next = np.linalg.pinv(Lk) @ Tk @ A[k+1, :k+1].T
        l_hat_next = np.linalg.pinv(Lk) @ A[:k+1, k+1].T

        s = A[k+1, k+1] - l_hat_next.T @ Tk @ l_hat_next

        t = 1 if s > 0 else -1
        
        gamma = t * s

        L[k+1, k+1] = np.sqrt(gamma)

        # t_next = L_k^{-T} @ (l_next - t * l_hat_next)
        T[k+1, :k+1] = np.linalg.pinv(Lk).T @ (l_next - t * l_hat_next).T

    return L, T
    
        
    
    # Compute S = L @ L.T
    #S = L @ L.T
    
    # Solve A = S @ T
    #A_computed = S @ T
    
    #return L, T, A_computed

# Example usage:
A = np.array([[4, 2, 1],
              [2, 2, 1],
              [0, 1, 1]])

print(np.linalg.det(A))

# check if A is symmetric
print(np.allclose(A, A.T))

l, t = golub_yuan_st_decomposition(A)

print(l)

print(t)

s = l @ l.T

print(s @ t)

'''print("L matrix:")
print(L)
print("\nT matrix:")
print(T)
print("\nComputed A matrix:")
print(A_computed)'''
