#!/usr/bin/env python3

from tools.parseData import parse_data
from settings import *

def parseData():
    parse_data(TRAIN_DATA_SET)
    parse_data(TEST_DATA_SET)

if __name__ == '__main__':
    if ALGORITHM == BRUTE_ALGORITHM:
        from core.brute import runBruteTest
        res = runBruteTest()
    if ALGORITHM == MIN_MAX_ALGORITHM:
        from core.minmax import runMinMaxTest
        res = runMinMaxTest()
    if ALGORITHM == PARALLELIZED_MIN_MAX:
        from core.multiProc import runMultiProcessMinMaxTest
        res = runMultiProcessMinMaxTest()
    if res != None:
        print(res)
