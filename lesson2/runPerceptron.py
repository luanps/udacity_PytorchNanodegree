import sys
import pdb
import numpy as np
from plotRes import plotData
from perceptron import trainPerceptronAlgorithm
X = []
Y = []
with open(sys.argv[1],'r') as f:
    for line in f.readlines():
        li = line.split('\n')[0]
        x1,x2,y = li.split(',')
        X.append([float(x1),float(x2)])
        Y.append(int(y))
res = trainPerceptronAlgorithm(X,Y,0.01,25)
plotData(X,Y,res)
