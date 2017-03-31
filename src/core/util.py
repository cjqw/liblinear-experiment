from liblinearutil import *

def getModel(y,x,cmd):
    return train(y,x,cmd)

def predictResult(y,x,model):
    return predict(y,x,model)
