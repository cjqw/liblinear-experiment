from tools.tools import *
from utils.util import *
import tools.dataIO as IO
from tools.tools import *
from utils.util import *
from timer.core import Timer
import logging

def runBruteTest(timer):
    train_timer = timer.start("train")
    data_set = IO.read_data(TRAIN_DATA_SET)
    test_set = IO.read_data(TEST_DATA_SET)
    logging.info("Begin to get brute model...")
    m = getModel(data_set,'Brute.model')
    timer.end(train_timer)

    logging.info("Begin to test brute algorithm...")
    test_timer = timer.start("test")
    result = predictResult(test_set, m)
    result = compareResult(result,mapv(getValue("sign"),test_set))
    timer.end(test_timer)
    IO.dump_data(result['ROC'],MODEL_NAME)
    return result
