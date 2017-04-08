#!env /usr/bin/python3

from tools.readData import read_data
from tools.parseData import parse_data
from core.brute import runBruteTest
from core.minmax import runMinMaxTest
from settings import *

if __name__ == '__main__':
    # parse_data(TRAIN_DATA_SET)
    # parse_data(TEST_DATA_SET)

    data = read_data(TRAIN_DATA_SET)
    test = read_data(TEST_DATA_SET)
    res = runBruteTest(data,test)
    # res = runMinMaxTest(data,test)
    print(res["acc"])
