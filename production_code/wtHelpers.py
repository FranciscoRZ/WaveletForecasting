import math
import numpy as np

def __compute_max_j(delta_j: float, T: int, a0: float) -> float:
    '''page 9 in article
    '''
    return pow(delta_j, -1.0) * np.log2(T / a0) + 1

def __compute_aj(delta_j: float, j: float) -> float:
    ''' page 9 in article
    '''
    return pow(2, 1 + j * delta_j)

def compute_scaling_parameter_grid(delta_j: float, a0: float, T: int) -> list:
    ''' page 9 in article
    ''' 
    all_scales = list()
    J = __compute_max_j(delta_j, T, a0)
    for j in range(0, int(J)+1):
        all_scales.append(__compute_aj(delta_j, j))
    
    return all_scales