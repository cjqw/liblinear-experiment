#!/usr/bin/env python3

from utils.util import *
import matplotlib.pyplot as plt

def drawROC(dots, name):
    tpr = mapv(lambda x: x[1], dots)
    fpr = mapv(lambda x: x[2], dots)
    plt.plot(fpr,tpr,'r-',label = name)
    plt.legend()
    plt.show()
