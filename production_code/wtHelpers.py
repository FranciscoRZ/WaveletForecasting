import math
import numpy as np

def computeAj(a0: float, delta_j: float, j: float) -> float:
    return a0 * pow(2, j * delta_j)
    
def computeJ(a0: float, delta_j: float, T: int) -> int:
    return math.floor(pow(delta_j, -1) * np.log2(T / a0)) + 1
    
def computeScaleGrid(a0: float, delta_j: float, T: int) -> list:
    J = computeJ(a0, delta_j, T)
    return [computeAj(a0, delta_j, j) for j in range(J)]
