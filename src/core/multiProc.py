from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Pool
import tools.dataIO as IO
from tools.readData import read_data

def store2File(m, name):
    for key in m:
        IO.save_data(m[key],'TMP' + str(key) + name)
        m[key] = None
    return m

def calcModel(i,j,fileName):
    print('START',str(i),str(j))
    x = IO.read_data('TMP' + str(i) + 'POS')
    y = IO.read_data('TMP' + str(j) + 'NEG')
    getModel(x + y, '-c 4', fileName)

def multiProcessTrainFunc(pos,neg,nameFunc):
    p = Pool()
    for i in pos:
        for j in neg:
            p.apply_async(calcModel, args = (i,j,nameFunc(i,j)))
    p.close()
    p.join()

def multiProcessPredictResult(model,i,j):
    print ('START')
    test = read_data(TEST_DATA_SET)
    res = predictResult(test,model)
    IO.save_data(res,str(i) + '|' + str(j) + '.result')

def multiProcessGetResult(models):
    p = Pool()
    for i in models:
        for j in models[i]:
            m = models[i][j]
            p.apply_async(multiProcessPredictResult,
                    args = (m,i,j))
    p.close()
    p.join()
    result = mapValue(partial(mapValue,constant(None)),
                      models)
    for i in result:
        for j in result[i]:
            result[i][j] = IO.read_data(str(i) + '|' + str(j) + '.result')
    return result

def runMultiProcessMinMaxTest():
    data = read_data(TRAIN_DATA_SET)
    print('Begin to get multi-process min-max model...')
    pos,neg = partitionData(data,getRandClass)
    neg = store2File(neg,'NEG')
    pos = store2File(pos,'POS')
    data = None

    models = getMinMaxModels(pos,
                             neg,
                             modelNameFunc('MultiProcessMinMax'),
                             multiProcessTrainFunc)

    print('Begin to test multi-process min-max algorithm...')
    res = multiProcessGetResult(models)
    test = read_data(TEST_DATA_SET)
    return minMaxPredictResult(test,res)
