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
    return mapv(lambda x:x[0],p_val)

def update(item,predict,answer):
    # th,fp,tp,fn,tn = item
    if predict > item[0]:
        k = (answer + 3) >> 1
    else:
        k = (7 - answer) >> 1
    item[k] = item[k] + 1

def calcROC(item):
    th,fp,tp,fn,tn = item
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    return [th,tpr,fpr]

def compareResult(predict,answer):
    thresholds = [-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8]
    thresholds = mapv(lambda x: [x,0,0,0,0],thresholds)

    for i in range(0,len(answer)):
        for items in thresholds :
            update(items,predict[i],answer[i])

    th,fp,tp,fn,tn = thresholds[5]
    r = tp/(tp + fn)
    p = tp/(tp + fp)
    return {
        "acc": ((tp + tn)*100)/len(answer),
        'f1': (2*r*p)/(r+p),
        'ROC': mapv(calcROC,thresholds)
    }
