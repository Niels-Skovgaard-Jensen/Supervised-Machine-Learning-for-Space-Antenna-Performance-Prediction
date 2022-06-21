import torch
from pathlib import Path
import yaml


def getSaveModeldir(extra_step_back = 0):
    main_dir = Path().cwd().parents[1+extra_step_back]
    models_dir = main_dir / 'models' 

    return models_dir

def saveConfig(config,name, subfolder = None):
    name = name +'.yaml'
    models_dir = getSaveModeldir()
    if type(subfolder) is type(str()):
        PATH = models_dir / subfolder / name
    else:
        PATH = models_dir / name
        
    with open(PATH, 'w') as stream:
        yaml.dump(config,stream)


def saveModel(model,name, subfolder = None,extra_step_back=0):
    assert type(name) == type(str())
    assert type(subfolder) == type(str()) or type(subfolder) == type(None)

    name = name +'.pt'
    models_dir = getSaveModeldir(extra_step_back = extra_step_back)
    if type(subfolder) is type(str()):
        PATH = models_dir / subfolder / name
    else:
        PATH = models_dir / name
        
    torch.save(model.state_dict(), PATH)

    return True

def loadModel(model,name, subfolder = None):
    assert type(name) == type(str())

    models_dir = getSaveModeldir()
    if subfolder is None:
        PATH = models_dir / name
    else:
        PATH = models_dir / subfolder / name
    print(PATH)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(PATH,map_location=torch.device('cuda:0')))
    else:
        model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    return model

