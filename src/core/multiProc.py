from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Pool
import tools.dataIO as IO

def store2File(key,value,add = ''):
    IO.save_data(value,'TMP' + str(key) + add)

def calcModel(i,j,nameFunc):
    print('START',str(i),str(j))
    y = IO.read_data('TMP' + str(j))
    x = IO.read_data('TMP' + str(i) + 'pos')
    getModel(x + y, '-c 4', nameFunc(i,j))
    print('OK')

def multiProcessTrainFunc(pos,neg,nameFunc):
    p = Pool(3)
    for i in pos:
        for j in neg:
            p.apply_async(Process(calcModel, args = (i,j,nameFunc)))
    p.close()
    p.join()

def runMultiProcessMinMaxTest(data,test):
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
    return minMaxPredictResult(test, models)
