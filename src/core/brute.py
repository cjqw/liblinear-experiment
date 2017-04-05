from tools.tools import *
from utils.util import *


def runBruteTest(data,test):
    print("Begin to get brute model...")
    m = getModel(data, '-c 4','Brute.model')
    print("Begin to test brute algorithm...")
    p_label, p_acc, p_val = predictResult(test, m)
    return {"label":p_label,
            "acc":p_acc}
