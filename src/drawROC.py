#!/usr/bin/env python3
from utils.util import *
import tools.dataIO as IO
import matplotlib.pyplot as plt

labels = ["Brute Algorithm",
              "Random Min-Max",
              "Labeled Min-Max"]
for label in labels:
    dots = IO.load_data(label)
    tpr = mapv(lambda x: x[1], dots)
    fpr = mapv(lambda x: x[2], dots)
    plt.plot(fpr,tpr,label = label)

plt.legend()
plt.show()
