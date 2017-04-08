from random import shuffle
from utils.util import *
from settings import *
import json

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

def writeAsJSON(data,fileName):
    with open(DATA_PATH + fileName , "w") as fout:
        json.dump(data,fout)

def parse_data(input_file):
    """Parse the data into json format and store them with '.json' suffix."""
    with open(DATA_PATH + input_file , "r")  as fin:
        data = fin.readlines()
    result = mapv(parse_lines,data)
    mapv(calc_sign,result)
    writeAsJSON(result,input_file + '.json')
