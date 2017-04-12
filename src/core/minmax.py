from tools.tools import *
from utils.util import *
from random import randint

def getRandClass(m):
    return randint(1,MAX_CLASS) - 1

def modelNameFunc(Name):
    def modelNameFunction(i,j):
        return Name + '[' + str(i) + ']-[' + str(j) + '].model'
    return modelNameFunction

def partitionData(data,partitionFunc):
    # partition data set by sign
    m = partition(data,getValue("sign"))
    pos,neg = m[1],m[-1]
    # partition each data set via partitionFunc
    pos = partition(pos,partitionFunc)
    neg = partition(neg,partitionFunc)
    return pos,neg

def bruteTrainFunc(pos,neg,nameFunc):
    for i in pos:
        for j in neg:
            getModel(pos[i]+neg[j],'-c 4',nameFunc(i,j))

def getMinMaxModels(pos,neg,nameFunc,trainFunc):
    trainFunc(pos,neg,nameFunc)
    models = {}
    for i in pos:
        models.update( {i : {}} )
        for j in neg:
            models[i].update(
                {j : loadModel(nameFunc(i,j))}
            )
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
    return float(x) * float(y) > 0

def minMaxPredictResult(test,models):
    result = mapValue(partial(mapValue,partial(predictResult,test)),models)
    result = mapValue(partial(minMaxLayer,min),result)
    result = minMaxLayer(max,result)
    acc = reduce(add,map(equal,result["label"],map(getValue("sign"),test)))
    result.update({"acc": acc * 100 / len(test)})
    return result

def runMinMaxTest(data,test):
    print('Begin to get min-max model...')
    pos,neg = partitionData(data,getRandClass)
    models = getMinMaxModels(pos,
                             neg,
                             modelNameFunc('MinMax'),
                             bruteTrainFunc)
    print('Begin to test min-max algorithm...')
    return minMaxPredictResult(test, models)
