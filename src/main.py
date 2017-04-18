#!/usr/bin/env python3

from tools.parseData import parse_data
from settings import *
from timer.core import Timer

def parseData():
    parse_data(TRAIN_DATA_SET)
    parse_data(TEST_DATA_SET)

timer = Timer("total","train","test")

def run():
    global timer
    total_timer = timer.start("total")
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

    timer.end(total_timer)

if __name__ == '__main__':
    run()
    timer.output()
