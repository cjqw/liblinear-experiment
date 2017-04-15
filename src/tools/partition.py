from utils.util import *

def getRandClass(item):
    return randint(1,MAX_CLASS) - 1

def partitionByFirstLetter(item):
    return item["label"][0][0]

def partitionByMiddleNumber(item):
    s = item["label"][0].split('/')
    return s[1]

def partitionByLastLetter(item):
    s = item["label"][0].split('/')
    return s[0][-1:]
