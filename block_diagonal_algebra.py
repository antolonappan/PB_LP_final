''' Block diagonal algebra

Perform algebraic operations over the last two axes of numpy arrays.

Most of the functions emulate @ in python 3 or np.matmul in python 2, adding
the possibility of weighting in matrix-matrix, or matrix-vector operations.
'''
import numpy as np

def inv(m):
    result = np.array(map(np.linalg.inv, m.reshape((-1,)+m.shape[-2:])))
    return result.reshape(m.shape)

def mv(m, v):
    return np.einsum('...ij,...j->...i', m, v)

def mtv(m, v):
    return np.einsum('...ji,...j->...i', m, v)

def mm(m, n):
    return np.einsum('...ij,...jk->...ik', m, n)

def mtm(m, n):
    return np.einsum('...ji,...jk->...ik', m, n)

def wv(w, v):
    return np.einsum('...i,...i->...i', w, v)

def wm(w, m):
    return np.einsum('...j,...ji->...ji', w, m)

def mwv(m, w, v):
    return np.einsum('...ij,...j,...j->...i', m, w, v)

def mtwv(m, w, v):
    if len(w.shape) == 1 and len(m.shape) == 2: # 8X speedup
        wmt = (m * w[:,np.newaxis]).T
        return np.einsum('...ij,...j->...i', wmt, v)
    return np.einsum('...ji,...j,...j->...i', m, w, v)

def mwm(m, w, n):
    return np.einsum('...ij,...j,...jk->...ik', m, w, n)

def mtwm(m, w, n):
    return np.einsum('...ji,...j,...jk->...ik', m, w, n)

def T(x):
    # Indexes < -2 are assumed to count diagonal blocks. Therefore the transpose
    # has to swap the last two axis, not reverse the order of all the axis
    try:
        return np.swapaxes(x, -1, -2)
    except ValueError:
        return x
