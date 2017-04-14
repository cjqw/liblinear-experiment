from utils.util import *

def getRandClass(item):
    return randint(1,MAX_CLASS) - 1

def partitionByFirstLabel(item):
    return item["label"][0][0]
