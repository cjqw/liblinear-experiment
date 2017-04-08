from random import shuffle
from utils.util import *
from settings import *
import json

def parseParam(item):
    params = {}
    m = item['param']
    for key in m:
        params.update(
            {int(key):m[key]})
    item.update(
        {'param':params})
    return item

def read_data(input_file):
    """Read data from json file. If TRAIN_DATA is greater than 0,
    pick TRAIN_DATA items from the input randomly."""
    with open(DATA_PATH + input_file + '.json', 'r') as fin:
        data = json.load(fin)
    if TRAIN_DATA > 0:
        shuffle(data)
        data = data[:TRAIN_DATA]
        print("The data has been shuffled...")
    return mapv(parseParam,data)
