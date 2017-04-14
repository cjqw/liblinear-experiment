from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Pool
import tools.dataIO as IO
from tools.readData import read_data
from time import time

MPModelName = metaNameFunc('MultiProcessMinMax','model')
MPPOSName = metaNameFunc('MultiProcessMinMax','POS')
MPNEGName = metaNameFunc('MultiProcessMinMax','NEG')
MPResultName = metaNameFunc('MultiProcessMinMax','result')
test_set = None
neg = None
pos = None

def store2File(m, nameFunc):
    for key in m:
        IO.save_data(m[key], nameFunc(key))
    return mapValue(constant(None),m)

def calcModel(param):
    i,j,fileName = param
    x = IO.read_data(MPPOSName(i))
    y = IO.read_data(MPNEGName(j))
    getModel(x + y, '-c 4', fileName)

def multiProcessTrainFunc(pos,neg,nameFunc):
    params = [[i,j,nameFunc(i,j)] for j in neg for i in pos]
    with Pool() as p:
        p.map(calcModel,params)

def multiProcessPredictResult(posLabel):
    global test_set,neg
    result = sequence(len(test_set),constant(1))
    for negLabel in neg:
        model = loadModel(MPModelName(posLabel,negLabel))
        result = mapv(min,result,predictResult(test_set,model)["label"])
    IO.save_data(result,MPResultName(posLabel))

def multiProcessGetResult():
    global test_set,pos,neg
    params = [posLabel for posLabel in pos]
    with Pool() as p:
        p.map(multiProcessPredictResult,params)

    result = sequence(len(test_set),constant(-1))
    for i in pos:
        result = mapv(max,result,IO.read_data(MPResultName(i)))

    acc = reduce(add,
                 map(equal,result,map(getValue("sign"),test_set)))

    return {
        "acc": (acc * 100 / len(test_set)),
        "label" : result
    }


def runMultiProcessMinMaxTest():
    global test_set,pos,neg
    data = read_data(TRAIN_DATA_SET)
    print('Begin to get multi-process min-max model...')
    pos,neg = partitionData(data,getRandClass)
    neg = store2File(neg,MPNEGName)
    pos = store2File(pos,MPPOSName)
    models = mapValue(constant(neg),pos)
    data = None

    getMinMaxModels(pos,neg,MPModelName,multiProcessTrainFunc)
    print('Begin to test multi-process min-max algorithm...')
    test_set = read_data(TEST_DATA_SET)
    result = multiProcessGetResult()
    return result
