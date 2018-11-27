import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plotData(X,Y,res):
    x0 = np.asarray(X)[:,0]
    x1 = np.asarray(X)[:,1]
    plt.scatter(x0,x1,c=Y)
    for i in res:
        plt.plot(i)
    plt.xlim(-.5,1.5)
    plt.ylim(-.5,1.5)
    plt.show()
