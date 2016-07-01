import numpy as np

"""
Simple implementation of Lanczos from pseudocode given on wikipedia page
(https://en.wikipedia.org/wiki/Lanczos_algorithm). Only reliable for the ground
state. Should check residual to make sure what is returned is an eigenvector.
"""

def Lanczos(A,k,v,dtype=np.float,return_eigenvectors=True, low_mem=False):
    """
    Lanczos implementation from wikipedia pseudocode

    Parameters
    ----------
    A: function ref
        Function which multiplies matrix
    k: int
        Number of requiested eigenvalues
    v: vector
        The starting vector
    eigenvector: bool
        Flag to indicate whether to calculate eigenvectors or not (default=True).
    low_mem: bool
        Flag to indicate that keeping the memory requirements low is a priority
        and eigenvectors will be calculated in a second pass rather than storing
        intermediate vectors (default=False).

    Returns
    --------
    float
        Lowest eigenvalue
    vector
        Lowest eigenvector
    """
    n = len(v)
    if k > n: k = n
    # find elements of tri-diagonal matrix
    v0 = np.copy(v)/np.linalg.norm(v)
    alpha = np.zeros(k, dtype=dtype)  #diagonal
    beta = np.zeros(k, dtype=dtype)   #offdiagonal
    save_vectors = not low_mem and return_eigenvectors
    if save_vectors:
        vs = np.zeros((n,k), dtype=dtype)
        vs[:,0] = v0
    for j in range(k-1):
        omega = A(v0)
        alpha[j] = np.dot(omega.conj().T, v0)
        omega = omega -alpha[j]*v0 - (0.0 if j == 0 else beta[j]*v1)
        beta[j+1] = np.linalg.norm(omega)
        v1 = omega/beta[j+1]
        if save_vectors: vs[:,j+1] = v1
        v0, v1 = v1, v0
    omega = A(v0)
    alpha[k-1] = np.dot(omega.conj().T,v0)

    # Get lowest eigenvalue of tri-diagonal matrix
    T = np.vstack((beta, alpha))
    w, u = sp.linalg.eig_banded(T, select='i', select_range=(0,0))

    if not return_eigenvectors: # Don't calculate the eigenvector, just return.
        return w
    elif return_eigenvectors and low_mem: # Calculate eigenvectors by stepping back through everything.
        v0 = np.copy(v)/np.linalg.norm(v)
        f = np.zeros(n, dtype=dtype)
        for j in range(k-1):
            f += u[j] * v0
            omega = A(v0)
            omega = omega -alpha[j]*v0 - (0.0 if j == 0 else beta[j]*v1)
            v1 = omega/beta[j+1]
            v0, v1 = v1, v0
        f += u[k-1] * v0
        return [w, f]
    else:  # Calculate eigenvectors with saved intermediate vectors.
        f = np.zeros(n, dtype=dtype)
        for j in range(k):
            f += u[j] * vs[:,j]
        return [w, f]

