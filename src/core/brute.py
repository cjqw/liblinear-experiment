from tools.tools import *
from utils.util import *

def getlabel(x):
    label = x[0][0]
    if label[0] == 'A':
        return 1
    else:
        return -1

def runBruteTest(data,test):
    print("Begin to run brute algorithm...")
    y = mapv(getlabel,data)
    x = mapv(lambda x: x[1],data)
    m = getModel(y, x, '-c 4')
    print("Begin to test brute algorithm...")
    y = mapv(getlabel,test)
    x = mapv(lambda x:x[1],test)
    p_label, p_acc, p_val = predictResult(y, x, m)
