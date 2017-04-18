from tools.tools import *
from utils.util import *
from tools.dataIO import read_data
from timer.core import Timer



def runBruteTest():
    global timer
    train_timer = timer.start("train")
    data_set = read_data(TRAIN_DATA_SET)
    test_set = read_data(TEST_DATA_SET)
    print("Begin to get brute model...")
    m = getModel(data_set,'Brute.model')
    timer.end(train_timer)

    print("Begin to test_set brute algorithm...")
    test_timer = timer.start("test")
    result = predictResult(test_set, m)
    result = compareResult(result,mapv(getValue("sign"),test_set))
    timer.end(test_timer)

    return result
