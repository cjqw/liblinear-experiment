from utils.util import *
from settings import *
import pickle

def save_data(data,fileName):
    '''TODO add MEMORIZE logic'''
    with open(DATA_PATH + fileName + '.pickle', 'wb') as fout:
        pickle.dump(data,fout)

def read_data(fileName):
    with open(DATA_PATH + fileName + '.pickle', 'rb') as fin:
        data = pickle.load(fin)
    return data

def store2File(m, nameFunc):
    for key in m:
        save_data(m[key], nameFunc(key))
    return mapValue(constant(None),m)
