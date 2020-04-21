''' Haar wavelet transform implemented here as described in the book
Coding the Matrix: Linear Algebra Through Computer Science Applications, Philip N. Klein
First Edition.
'''
    
def __forward_no_normalization(v: list) -> dict:
    D = {}
    while (len(v) > 1):
        # v is a k-element list
        k = len(v)
        # vnew is a k // 2 -element list
        vnew = [(v[2*i] + v[2*i + 1]) / 2 for i in range(0, k // 2)]
        # w is a list of coefficients of lenght k // 2
        w = [(v[2*i] - v[2*i + 1]) for i in range(0, k // 2)]  # unnormalized coefficients of basis for W(k/2)
        # dictionary with keys (k//2, 0), (k//2, 1), (k//2, k//2-1) and values from w
        D.update({(k//2, i) : w[i] for i in range(len(w))}) 
        v = vnew
    # v is a 1-element list
    D[(0,0)] = v[0] # store the last wavelet coefficient
    return D
    
def __normalize_coefficients(n: int, coeffs: dict) -> dict:
    keys = [x for x in coeffs.keys() if x != (0,0)]
    D = dict()
    for si in keys:
        norm = pow(n / (4 * si[0]), 0.5)
        D.update({si:coeffs[si] * norm})
    D.update({(0,0):coeffs[(0,0)] * pow(n, 0.5)})
    return D

def forward(v: list) -> dict:
    n = len(v)
    return normalize_coefficients(n, forward_no_normalization(v))

def suppress(D: dict, threshold: float) -> dict:
    return {k:(v if abs(v) > threshold else 0.0) for (k, v) in D.items()}

def sparsity(D: dict) -> float:
    num_non_zeros = sum([1 if abs(x) > 0.00001 else 0 for x in D.values()])
    return num_non_zeros / len(D)

def __unnormalize_coefficients(n: int, D: dict) -> dict:
    keys = [k for k in D.keys() if k != (0,0)]
    coeffs = dict()
    for si in keys:
        norm = pow(n / (4 * si[0]), 0.5)
        coeffs.update({si:D[si] / norm})
    coeffs.update({(0,0): D[(0,0)] / pow(n, 0.5)})
    return coeffs

def __backward_no_normalization(D: dict) -> list:
    n = len(D)
    v = [D[(0,0)]]
    while len(v) < n:
        b_ix = len(v)
        vnew = list()
        for w_ix, w in enumerate(v):
            w_val = D[(b_ix, w_ix)]
            vnew.append(w + (w_val / 2))
            vnew.append(w - (w_val / 2))
        v = vnew
    return v

def backwards(D: dict) -> list:
    n = len(D)
    return backward_no_normalization(unnormalize_coefficients(n, D))