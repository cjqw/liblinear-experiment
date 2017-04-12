#!/usr/bin/env python3
# POS NEG
# 27876 85252


from tools.parseData import parse_data
from core.brute import runBruteTest
from core.minmax import runMinMaxTest
from core.multiProc import runMultiProcessMinMaxTest
from settings import *

if __name__ == '__main__':
    # parse_data(TRAIN_DATA_SET)
    # parse_data(TEST_DATA_SET)
    # res = runBruteTest()
    res = runMinMaxTest()
    # res = runMultiProcessMinMaxTest()
    print(res["acc"])
