from utils.util import *
from random import randint

def getRandClassFunc(max_class):
    def getRandClass(item):
        return randint(1,max_class) - 1
    return getRandClass

def getFirstLetter(item):
    return item["label"][0][0]

def getFirstTwoLetter(item):
    return item["label"][0][:2]

def getLastLetter(item):
    s = item["label"][0].split('/')
    return s[0][-1:]
