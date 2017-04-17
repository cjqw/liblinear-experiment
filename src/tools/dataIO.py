from liblinearutil import *
from utils.util import *
from settings import *
import pickle

def dump_data(data,fileName):
    '''TODO add MEMORIZE logic'''
    with open(DATA_PATH + fileName + '.pickle', 'wb') as fout:
        pickle.dump(data,fout)

def load_data(fileName):
    with open(DATA_PATH + fileName + '.pickle', 'rb') as fin:
        data = pickle.load(fin)
    return data

def store2File(m, nameFunc):
    for key in m:
        dump_data(m[key], nameFunc(key))
    return mapValue(constant(None),m)

def read_data(input_file):
    """Read data from json file. If TRAIN_DATA is greater than 0,
    pick TRAIN_DATA items from the input randomly."""
    with open(DATA_PATH + input_file + '.pickle', 'rb') as fin:
        data = pickle.load(fin)
    if TRAIN_DATA > 0:
        shuffle(data)
        data = data[:TRAIN_DATA]
        print("The data has been shuffled...")
    return data

def loadModel(name):
    with cd(MODEL_PATH):
        return load_model(name)

def saveModel(m,name):
    with cd(MODEL_PATH):
        save_model(name,m)
