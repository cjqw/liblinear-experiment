from tools.tools import *
from tools.partition import *
from utils.util import *
from random import randint
import tools.dataIO as IO

def partitionData(data,partitionFunc):
    m = partition(data,getValue("sign"))
    pos,neg = m[1],m[-1]
    pos = partition(pos,partitionFunc)
    neg = partition(neg,partitionFunc)
    return pos,neg

def bruteTrainFunc(pos,neg,nameFunc):
    for i in pos:
        for j in neg:
            getModel(pos[i]+neg[j],nameFunc(i,j))

def getMinMaxModels(pos,neg,nameFunc,trainFunc):
    trainFunc(pos,neg,nameFunc)
    models = {}
    for i in pos:
        models.update( {i : {}} )
        for j in neg:
            models[i].update(
                {j : IO.loadModel(nameFunc(i,j))}
            )
    return models

def minMaxPredictResult(test_set,models):
    length = len(test_set)
    result = sequence(length,constant(-1))
    for p in models:
        t = sequence(length,constant(1))
        for n in models[p]:
            k = predictResult(test_set,models[p][n])
            t = mapv(min,t,k)
        result = mapv(max,t,result)

    return compareResult(result,mapv(getValue("sign"),test_set))


def runMinMaxTest(timer):
    train_timer = timer.start("train")
    data = IO.read_data(TRAIN_DATA_SET)
    test = IO.read_data(TEST_DATA_SET)
    print('Begin to get min-max model...')
    pos,neg = partitionData(data,PARTITION_FUNCTION)
    models = getMinMaxModels(pos,
                             neg,
                             metaNameFunc('MinMax','model'),
                             bruteTrainFunc)
    timer.end(train_timer)
    print('Begin to test min-max algorithm...')
    test_timer = timer.start("test")
    result =  minMaxPredictResult(test,models)
    timer.end(test_timer)
    return result
