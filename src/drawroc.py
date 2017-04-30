#!/usr/bin/env python3
from utils.util import *
import tools.dataIO as IO
import matplotlib.pyplot as plt

labels = ["Brute Algorithm",
          "Random Min-Max",
          "Labeled Min-Max"]

def calcAUC(X,Y,name):
    X = [1] + X + [0]
    Y = [1] + Y + [0]
    X = mapv(lambda x: 1-x,X)
    print(name,":")
    px,py = 0,0
    s = 0
    for (x,y) in zip(X,Y):
        s = s + px * y - py * x
        px,py = x,y
    print(abs(s)/2)

for label in labels:
    dots = IO.load_data(label)
    tpr = mapv(lambda x: x[1], dots)
    fpr = mapv(lambda x: x[2], dots)
    plt.plot(fpr,tpr,label = label)
    calcAUC(fpr,tpr,label)

plt.legend()
plt.show()
