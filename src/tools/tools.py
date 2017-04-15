from liblinearutil import *
from settings import *
from utils.util import *

def getModel(data,name = "MODEL",cmd = "-c 4",mem = MEMORIZE):
    y = mapv(getValue("sign"),data)
    x = mapv(getValue("param"),data)
    model = None
    if mem:
        with cd(MODEL_PATH): model = load_model(name)

    if model == None:
        model = train(y,x,cmd)

    with cd(MODEL_PATH):
        save_model(name,model)

    return model

def predictResult(data ,model):
    y = mapv(getValue("sign"),data)
    x = mapv(getValue("param"),data)
    p_label, p_acc, p_val = predict(y,x,model)
    return{"label" : p_label,
           "acc" : p_acc[0],
           "val" : p_val}

def loadModel(name):
    with cd(MODEL_PATH):
        return load_model(name)
