from tools.tools import *
from utils.util import *
from random import randint

def getRandomClass(m):
    return randint(1,MAX_CLASS)-1

def runMinMaxTest(data,test):
    data = partition(data,getRandomClass)
    print("Begin to get min-max model...")
    # # m = getModel(data, '-c 4','Brute.model')
    # print("Begin to test min-max algorithm...")
    # p_label, p_acc, p_val = predictResult(test, m)
    # return {"label":p_label,
    #         "acc":p_acc}
