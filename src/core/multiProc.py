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

def store2File(m, nameFunc):
    for key in m:
        IO.save_data(m[key], nameFunc(key))
        m[key] = None
    return m

def calcModel(i,j,fileName):
    x = IO.read_data(MPPOSName(i))
    y = IO.read_data(MPNEGName(j))
    getModel(x + y, '-c 4', fileName)

def multiProcessTrainFunc(pos,neg,nameFunc):
    p = Pool()
    for i in pos:
        for j in neg:
            p.apply_async(calcModel, args = (i,j,nameFunc(i,j)))
    p.close()
    p.join()

def multiProcessPredictResult(i,j):
    test = read_data(TEST_DATA_SET)
    model = loadModel(MPModelName(i,j))
    res = predictResult(test,model)
    IO.save_data(res,MPResultName(i,j))

def multiProcessGetResult(pos,neg):
    p = Pool()
    for i in pos:
        for j in neg:
            p.apply_async(multiProcessPredictResult,
                    args = (i,j))
    p.close()
    p.join()

    result = mapValue(constant(neg),pos)
    for i in result:
        for j in result[i]:
            result[i][j] = IO.read_data(MPResultName(i,j))
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

    res = multiProcessGetResult(pos,neg)
    test = read_data(TEST_DATA_SET)
    return minMaxPredictResult(test,res)
