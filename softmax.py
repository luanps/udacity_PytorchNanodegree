import numpy as np
L = [5,6,7] #example input

result = []
for i in range(len(L)):
    result.append(np.exp(L[i])/sum(np.exp(L)))
print(result)
