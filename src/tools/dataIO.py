from utils.util import *
from settings import *
import pickle

def save_data(data,fileName):
    with open(DATA_PATH + fileName + '.pickle', 'wb') as fout:
        pickle.dump(data,fout)

def read_data(fileName):
    with open(DATA_PATH + fileName + '.pickle', 'rb') as fin:
        data = pickle.load(fin)
    return data
