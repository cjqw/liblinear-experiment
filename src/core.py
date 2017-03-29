#!env /usr/bin/python3

from liblinearutil import *
from tools.readdata import *
from tools.util import *
from random import shuffle

def getlabel(x):
    label = x[0][0]
    if label[0] == 'A':
        return 1
    else:
        return -1

data = read_data()
# print(data[2])
shuffle(data)
y = mapv(getlabel,data)
x = mapv(lambda x: x[1],data)
m = train(y[:200], x[:200], '-c 4')
# print(x,y)
p_label, p_acc, p_val = predict(y[200:], x[200:], m)
