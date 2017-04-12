from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Pool
import tools.dataIO as IO
from tools.readData import read_data

def store2File(key,value,add = ''):
    IO.save_data(value,'TMP' + str(key) + add)

def calcModel(i,j,fileName):
    print('START',str(i),str(j))
    y = IO.read_data('TMP' + str(j))
    x = IO.read_data('TMP' + str(i) + 'pos')
    getModel(x + y, '-c 4', fileName)
    print('OK')

def multiProcessTrainFunc(pos,neg,nameFunc):
    p = Pool()
    for i in pos:
        for j in neg:
            print(i,j)
            p.apply_async(calcModel, args = (i,j,nameFunc(i,j)))
    p.close()
    p.join()

def runMultiProcessMinMaxTest():
    data = read_data(TRAIN_DATA_SET)
    print('Begin to get multi-process min-max model...')
    pos,neg = partitionData(data,getRandClass)
    for key in neg:
        store2File(key,neg[key])
        neg[key] = None
    for key in pos:
        store2File(key,pos[key],'pos')
        pos[key] = None
    data = None
    models = getMinMaxModels(pos,
                             neg,
                             modelNameFunc('MultiProcessMinMax'),
                             multiProcessTrainFunc)
    print('Begin to test multi-process min-max algorithm...')

    test = read_data(TEST_DATA_SET)
    return minMaxPredictResult(test, models)
