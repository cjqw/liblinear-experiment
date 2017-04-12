from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Pool
import tools.dataIO as IO
from tools.readData import read_data
from functools import partial

def store2File(m, name):
    for key in m:
        IO.save_data(m[key],'TMP' + str(key) + name)
        m[key] = None
    return m

def calcModel(x,neg,names):
    print('START')
    for j in neg:
        y = IO.read_data('TMP' + str(j) + 'NEG')
        getModel(x + y, '-c 4', names[j])
    print('OK')

def multiProcessTrainFunc(pos,neg,nameFunc):
    p = Pool()
    for i in pos:
        x = IO.read_data('TMP' + str(i) + 'POS')
        names = {}
        for j in neg: names.update({j:nameFunc(i,j)})
        p.apply_async(calcModel,args = (x,neg,names))
    p.close()
    p.join()

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
    return None
    print('Begin to test multi-process min-max algorithm...')
    test = read_data(TEST_DATA_SET)
    return minMaxPredictResult(test, models)
