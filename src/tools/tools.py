from liblinearutil import *
from settings import *
from utils.util import *
import tools.dataIO as IO

def metaNameFunc(name,suffix):
    def metaNameFunction(i,j = None):
        if j != None:
            return name + '[' + str(i) + ']-[' + str(j) + '].' + suffix
        else:
            return name + '[' + str(i) + '].' + suffix
    return metaNameFunction

def getModel(data,name = "MODEL",cmd = "-c 4",mem = MEMORIZE):
    y = mapv(getValue("sign"),data)
    x = mapv(getValue("param"),data)
    model = None
    if mem:
        model = IO.loadModel(name)
    if model == None:
        model = train(y,x,cmd)
    IO.saveModel(model,name)
    return model

def predictResult(data ,model):
    y = mapv(getValue("sign"),data)
    x = mapv(getValue("param"),data)
    p_label, p_acc, p_val = predict(y,x,model)
    return p_label

def compareResult(answer,predict):
    res = [0,0,0,0]
    for i in range(0,len(answer)):
        k = int(answer[i] + 1.5 + predict[i]/2)
        res[k] = res[k] + 1
    result = {
        'TP': res[3],
        'FP': res[1],
        'FN': res[2],
        'TN': res[0]
    }
    r = result['TP']/(result['TP'] + result['FN'])
    p = result['TP']/(result['TP'] + result['FP'])
    result.update({
        "acc": ((res[0] + res[3])*100)/len(answer),
        'f1': (2*r*p)/(r+p)
    })
    return result
