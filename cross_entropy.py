import numpy as np

# takes as input two lists Y, P 
#and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.asarray(Y)
    P = np.asarray(P)
    return -sum(Y * np.log(P) +  (1 - Y) * np.log(1 - P))
    
