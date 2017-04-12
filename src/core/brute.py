from tools.tools import *
from utils.util import *
from tools.readData import read_data

def runBruteTest(data,test):
    data = read_data(TRAIN_DATA_SET)
    test = read_data(TEST_DATA_SET)
    print("Begin to get brute model...")
    m = getModel(data, '-c 4','Brute.model')
    print("Begin to test brute algorithm...")
    return predictResult(test, m)
