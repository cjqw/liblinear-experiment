from tools.util import *

def parse_item(item):
    x,y = item.split(":")
    return [int(x),float(y)]

def parse_param(param):
    # m = {}
    # for item in param:
    #     m.set(1,1)
    # return m
    return mapv(parse_item,param)

def parse_lines(data_line):
    item = data_line[:-1].split(" ")
    return [item[0],parse_param(item[1:])]

def read_data():
    input_file = "data/train.txt"
    with open(input_file , "r")  as fin:
        data = fin.readlines()
    data = data[:10]
    return mapv(parse_lines,data)
