from utils.util import *

def getRandClass(item):
    return randint(1,MAX_CLASS) - 1

def getFirstLetter(item):
    return item["label"][0][0]

def getFirstTwoLetter(item):
    return item["label"][0][:2]

def getLastLetter(item):
    s = item["label"][0].split('/')
    return s[0][-1:]
