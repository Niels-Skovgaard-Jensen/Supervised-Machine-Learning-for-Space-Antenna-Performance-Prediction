import torch
from pathlib import Path



def getSaveModeldir():
    main_dir = Path().cwd().parents[1]
    models_dir = main_dir / 'models' 

    return models_dir

def saveModel(model,name):

    models_dir = getSaveModeldir()
    PATH = models_dir / name
    torch.save(model.state_dict(), PATH)

    return True

def loadModel(model,name):

    models_dir = getSaveModeldir()
    PATH = models_dir / name
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(PATH))
    else:
        model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    return model

