import numpy as np
from scipy import io

def svd_method_of_snapshots(X):
    # matriz de covariância
    #C = X.T @ X
    C = np.cov(X)
    
    # pegando os autovalores e autovetores da matriz de covariância
    eigvals, V = np.linalg.eigh(C)
    
    # ordenando
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]
    
    # calculando os valores singulares
    S = np.sqrt(eigvals)
    
    # fazendo U
    U = X @ V
    U = U / S
    
    return U, S, V.T

a = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

a = np.array(a)

u, s, vh = np.linalg.svd(a)

u_snap, s_snap, vh_snap = svd_method_of_snapshots(a)

a_rebuilt_normal = np.dot(u, np.dot(np.diag(s), vh))

a_rebuilt_snap = np.dot(u_snap, np.dot(np.diag(s_snap), vh_snap))

# set numpy to print 3 decimal points and without extra scientific notation
np.set_printoptions(precision=3, suppress=True)

print(a_rebuilt_normal)

print(a_rebuilt_snap)

'''yaleb = io.loadmat('C:/Users/danie/Documents/Unesp/ALCD/data/allFaces.mat')
yaleb.keys()

yaleb_faces = yaleb['faces']

U_yale, s_yale, Vh_yale = np.linalg.svd(yaleb_faces, full_matrices=False)

U_snap_yale, s_snap_yale, Vh_snap_yale = svd_method_of_snapshots(yaleb_faces)

print(U_yale[0])

print(U_snap_yale[0])'''