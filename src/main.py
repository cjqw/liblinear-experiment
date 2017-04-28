#!/usr/bin/env python3

from tools.parseData import parse_data
from settings import *
from timer.core import Timer
from core.brute import runBruteTest
from core.minmax import runMinMaxTest
from core.multiProc import runMultiProcessMinMaxTest

def parseData():
    parse_data(TRAIN_DATA_SET)
    parse_data(TEST_DATA_SET)

timer = Timer("total","train","test")

def run(timer):
    if PARSE_DATA:
        parseData()

    total_timer = timer.start("total")

    if ALGORITHM == BRUTE_ALGORITHM:
        res = runBruteTest(timer)

    if ALGORITHM == MIN_MAX_ALGORITHM:
        res = runMinMaxTest(timer)

    if ALGORITHM == PARALLELIZED_MIN_MAX:
        res = runMultiProcessMinMaxTest(timer)

    if res != None:
        print(res)

    timer.end(total_timer)

if __name__ == '__main__':
    run(timer)
    timer.output()
