from random import shuffle
from utils.util import *
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
    """To be finished."""
    return mapv(identity,labels.split(','))

def calc_sign(m):
    label = m["label"][0]
    if label[0] == 'A':
        sign = 1
    else:
        sign = -1
    m.update({"sign":sign})

def parse_lines(data_line):
    item = data_line[:-1].split(" ")
    return {"label":parse_label(item[0]),
            "param":parse_param(item[1:])}

def read_data(input_file):
    """Read data from input file. If TRAIN_DATA is greater than 0,
    pick TRAIN_DATA items from the input randomly."""
    with open(input_file , "r")  as fin:
        data = fin.readlines()
    if TRAIN_DATA > 0:
        shuffle(data)
        data = data[:TRAIN_DATA]
        print("The data has been shuffled...")
    result = mapv(parse_lines,data)
    mapv(calc_sign,result)
    return result
