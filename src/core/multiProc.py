from tools.tools import *
from utils.util import *
from core.minmax import *
from multiprocessing import Process
import tools.dataIO as IO

def calcModel(i,j,fileName):
    print(fileName)
    x = IO.read_data('POS' + str(i))
    y = IO.read_data('NEG' + str(j))
    getModel(x + y, '-c 4', fileName)
    print('OK')

def multiProcessTrainFunc(pos,neg,nameFunc):
    for i in pos:
        IO.save_data(pos[i],'POS' + str(i))
    for j in neg:
        IO.save_data(neg[j],'NEG' + str(j))

    p = []

    for i in pos:
        for j in neg:
            # p.append(Process(target = calcModel, args = (i,j,nameFunc(i,j))))
            calcModel(i,j,nameFunc(i,j))

    for task in p:
        task.start()
    for task in p:
        task.join()

def runMultiProcessMinMaxTest(data,test):
    print('Begin to get multi-process min-max model...')
    models = getMinMaxModels(data,
                             getRandClass,
                             modelNameFunc('MultiProcessMinMax'),
                             multiProcessTrainFunc)
    print('Begin to test multi-process min-max algorithm...')
    return minMaxPredictResult(test, models)
