from liblinearutil import *
from settings import *
from utils.util import *

def getModel(y,x,cmd,name = "MODEL"):
    with cd(MODEL_PATH):
        model = load_model(name)
        if model == None:
            model = train(y,x,cmd)
            save_model(name,model)
    return model

def predictResult(y,x,model):
    return predict(y,x,model)
