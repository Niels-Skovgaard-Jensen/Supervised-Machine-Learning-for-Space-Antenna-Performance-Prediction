from types import NoneType
import torch
from pathlib import Path



def getSaveModeldir():
    main_dir = Path().cwd().parents[1]
    models_dir = main_dir / 'models' 

    return models_dir

def saveModel(model,name, subfolder = None):
    assert type(model) == type(str())
    assert type(name) == type(str())
    assert type(subfolder) == type(str()) or type(subfolder) == type(None)

    models_dir = getSaveModeldir()
    if type(subfolder) is type(str()):
        PATH = models_dir / subfolder / name
    else:
        PATH = models_dir / name
        
    torch.save(model, PATH+'.pt')

    return True

def loadModel(name, subfolder = None):
    assert type(subfolder) == type(str())
    assert type(name) == type(str())

    models_dir = getSaveModeldir()
    if subfolder is None:
        PATH = models_dir / name
    else:
        models_dir / subfolder / name
    if torch.cuda.is_available():
        model = torch.load(PATH,map_location=torch.device('cuda:0'))
    else:
        model = torch.load(PATH,map_location=torch.device('cpu'))
    return model

