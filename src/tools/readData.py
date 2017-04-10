from random import shuffle
from utils.util import *
from settings import *
import pickle

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
