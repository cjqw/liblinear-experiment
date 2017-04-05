from liblinearutil import *
from settings import *
from utils.util import *

def getModel(data,cmd,name = "MODEL"):
    y = mapv(lambda x: x["sign"],data)
    x = mapv(lambda x: x["param"],data)
    with cd(MODEL_PATH):
        model = load_model(name)
        if model == None:
            model = train(y,x,cmd)
            save_model(name,model)
    return model

def predictResult(data ,model):
    y = mapv(lambda x: x["sign"],data)
    x = mapv(lambda x: x["param"],data)
    return predict(y,x,model)
