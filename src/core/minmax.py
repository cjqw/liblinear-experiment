from tools.tools import *
from utils.util import *
from random import randint

def getRandClass(m):
    return randint(1,MAX_CLASS) - 1

def modelNameFunc(Name):
    def modelNameFunction(i,j):
        return Name + '[' + str(i) + ']-[' + str(j) + '].model'
    return modelNameFunction

def getMinMaxModels(data,partitionFunction,nameFunc):
    # partition data set by sign
    m = partition(data,getValue("sign"))
    pos,neg = m[1],m[-1]
    # partition each data set randomly
    pos = partition(pos,partitionFunction)
    neg = partition(neg,partitionFunction)

    models = {}
    for label in pos:
        models.update({label:{}})

    for i in pos:
        x = pos[i]
        for j in neg:
            y = neg[j]
            (models[i]).update(
                {j : getModel(x + y,'-c 4',nameFunc(i,j), memorize = False)})
    return models

def minMaxLayer(f,m):
    result = []
    for key in m:
        label = m[key]["label"]
        if len(result) == 0:
            result = label
        else:
            result = mapv(f,result,label)
    return {"label" : result}

def equal(x,y):
    return float(x)*float(y) > 0

def minMaxPredictResult(test,models):
    result = mapValue(partial(mapValue,partial(predictResult,test)),models)
    result = mapValue(partial(minMaxLayer,min),result)
    result = minMaxLayer(max,result)
    acc = reduce(add,map(equal,result["label"],map(getValue("sign"),test)))
    result.update({"acc": acc * 100 / len(test)})
    return result

def runMinMaxTest(data,test):
    print('Begin to get min-max model...')
    models = getMinMaxModels(data,getRandClass,modelNameFunc('MinMax'))
    print('Begin to test min-max algorithm...')
    return minMaxPredictResult(test, models)
