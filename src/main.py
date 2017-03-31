#!env /usr/bin/python3

from core.readData import read_data
from core.brute import runBruteTest
from settings import *


if __name__ == '__main__':
    data = read_data(TRAIN_DATA_SET)
    test = read_data(TEST_DATA_SET)
    runBruteTest(data,test)
