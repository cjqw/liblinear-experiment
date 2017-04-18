from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Pool
import tools.dataIO as IO

MPModelName = metaNameFunc('MultiProcessMinMax','model')
MPPOSName = metaNameFunc('MultiProcessMinMax','POS')
MPNEGName = metaNameFunc('MultiProcessMinMax','NEG')
MPResultName = metaNameFunc('MultiProcessMinMax','result')
test_set = None
neg = None
pos = None
models = None

def multiProcessCalcModel(param):
    i,j,fileName = param
    x = IO.load_data(MPPOSName(i))
    y = IO.load_data(MPNEGName(j))
    getModel(x + y, fileName)

def multiProcessTrainFunc(pos,neg,nameFunc):
    params = [[i,j,nameFunc(i,j)] for j in neg for i in pos]
    with Pool() as p:
        p.map(multiProcessCalcModel,params)

def multiProcessPredictResult(param):
    global test_set
    posLabel,negLabel = param
    model = models[posLabel][negLabel]
    result = predictResult(test_set,model)
    IO.dump_data(result,MPResultName(posLabel,negLabel))

def multiProcessGetResult():
    global test_set,pos,neg
    params = [[i,j] for i in pos for j in neg]
    with Pool() as p:
        p.map(multiProcessPredictResult,params)

    result = sequence(len(test_set),constant(-1))
    for i in pos:
        tmp = sequence(len(test_set),constant(1))
        for j in neg:
            tmp = mapv(min,tmp,IO.load_data(MPResultName(i,j)))
        result = mapv(max,result,tmp)

    return compareResult(result,mapv(getValue("sign"),test_set))


def runMultiProcessMinMaxTest():
    global test_set,pos,neg,models,timer
    train_timer = timer.start("train")
    data = IO.read_data(TRAIN_DATA_SET)
    print('Begin to get multi-process min-max model...')
    pos,neg = partitionData(data,PARTITION_FUNCTION)
    neg = IO.store2File(neg,MPNEGName)
    pos = IO.store2File(pos,MPPOSName)
    data = None

    models = getMinMaxModels(pos,neg,MPModelName,
                             multiProcessTrainFunc)
    timer.end(train_timer)

    print('Begin to test multi-process min-max algorithm...')
    test_timer = timer.start("test")
    test_set = IO.read_data(TEST_DATA_SET)
    result = multiProcessGetResult()
    timer.end(test_timer)

    return result
