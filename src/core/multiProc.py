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
test = None
pos = None
neg = None

def store2File(m, nameFunc):
    for key in m:
        IO.save_data(m[key], nameFunc(key))
        m[key] = None
    return m

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
    i,j = param
    global test
    model = loadModel(MPModelName(i,j))
    res = predictResult(test,model)
    IO.save_data(res,MPResultName(i,j))

def multiProcessGetResult(pos,neg):
    params = [[i,j] for i in pos for j in neg]
    with Pool() as p:
        p.map(multiProcessPredictResult,params)

    result = mapValue(constant(neg),pos)
    for i in pos:
        for j in neg:
            result[i][j] = IO.read_data(MPResultName(i,j))


    global test
    result = mapValue(partial(minMaxLayer,min),result)
    result = minMaxLayer(max,result)
    acc = reduce(add,
                 map(equal,result["label"],map(getValue("sign"),test)))
    result.update({"acc": acc * 100 / len(test)})
    return result


def runMultiProcessMinMaxTest():
    data = read_data(TRAIN_DATA_SET)
    print('Begin to get multi-process min-max model...')
    pos,neg = partitionData(data,getRandClass)
    neg = store2File(neg,MPNEGName)
    pos = store2File(pos,MPPOSName)
    data = None

    models = getMinMaxModels(pos,
                             neg,
                             MPModelName,
                             multiProcessTrainFunc)
    print('Begin to test multi-process min-max algorithm...')
    global test
    test = read_data(TEST_DATA_SET)
    result = multiProcessGetResult(pos,neg)
    return result
