from random import shuffle
from tools.util import *
from settings import *

def parse_item(item):
    x,y = item.split(":")
    return {int(x):float(y)}

def parse_param(param):
    m = {}
    for item in param:
        m.update(parse_item(item))
    return m

def parse_label(labels):
    return mapv(identity,labels.split(','))

def parse_lines(data_line):
    item = data_line[:-1].split(" ")
    return [parse_label(item[0]),parse_param(item[1:])]

def read_data(input_file):
    """Read data from input file. If TRAIN_DATA is greater than 0,
    pick TRAIN_DATA items from the input randomly."""
    with open(input_file , "r")  as fin:
        data = fin.readlines()
    if TRAIN_DATA > 0:
        shuffle(data)
        data = data[:TRAIN_DATA]
        print("The data has been shuffled...")
    return mapv(parse_lines,data)
