import numpy as np

def gausskernel(shape = 1):
    # return lambda x,y : np.exp(-shape**2 * np.linalg.norm(x-y, axis = 1)**2)
    return lambda x,y : np.exp(-shape**2 * np.linalg.norm(x-y)**2)

def wendlandkernel():
    return
