from tools.tools import *
from tools.partition import *
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
partitionFunc = getRandClass
partitionFunc = partitionByFirstLabel
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

def multiProcessPredictResult(param):
    global test_set
    posLabel,negLabel = param
    model = loadModel(MPModelName(posLabel,negLabel))
    result = predictResult(test_set,model)["label"]
    IO.save_data(result,MPResultName(posLabel,negLabel))

def multiProcessGetResult():
    global test_set,pos,neg
    params = [[i,j] for i in pos for j in neg]
    with Pool() as p:
        p.map(multiProcessPredictResult,params)

    result = sequence(len(test_set),constant(-1))
    for i in pos:
        tmp = sequence(len(test_set),constant(1))
        for j in neg:
            tmp = mapv(min,tmp,IO.read_data(MPResultName(i,j)))
        result = mapv(max,result,tmp)

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
    pos,neg = partitionData(data,partitionFunc)
    neg = store2File(neg,MPNEGName)
    pos = store2File(pos,MPPOSName)
    data = None

    getMinMaxModels(pos,neg,MPModelName,multiProcessTrainFunc)
    print('Begin to test multi-process min-max algorithm...')
    test_set = read_data(TEST_DATA_SET)
    result = multiProcessGetResult()
    return result
