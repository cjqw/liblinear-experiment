from tools.tools import *
from utils.util import *
from tools.dataIO import read_data

def runBruteTest():
    data_set = read_data(TRAIN_DATA_SET)
    test_set = read_data(TEST_DATA_SET)
    print("Begin to get brute model...")
    m = getModel(data_set,'Brute.model')
    print("Begin to test_set brute algorithm...")
    result = predictResult(test_set, m)
    return compareResult(result,mapv(getValue("sign"),test_set))
