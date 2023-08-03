# The Packages/Methods which are necessary for Model Loading phase are imported here.
from src.utils import Utility
from src.logger import logging
from src.components.model_trainer import ModelTrainer

# All operations related to Model Loading phase are carried out inside the class 'ModelLoader'
class ModelLoader:
    '''
    The 'loader_y1' function first tries to load the model in particular loaction. If not available in that location, 
    the trainer method of class ModelTrainer is called here.
    '''
    def loader_y1():
        try:
            model_y1 = Utility.load('./data/model/model_y1.pkl')
            logging.info("[model_loader.py] Model for Y1 is already there. So, loaded from default location")
        except:
            try:
                ModelTrainer.trainer_y1()
                model_y1 = Utility.load('./data/model/model_y1.pkl')
            except:
                logging.info("[model_loader.py] Somewhere the problem occurs, please check!!!")
            logging.info("[model_loader.py] Model is not present is the default location. So, we're going to develop a model")
        return model_y1

    '''
    The 'loader_y2' function first tries to load the model in particular loaction. If not available in that location, 
    the trainer method of class ModelTrainer is called here.
    '''
    def loader_y2():
        try:
            model_y2 = Utility.load('./data/model/model_y2.pkl')
            logging.info("[model_loader.py] Model for Y2 is already there. So, loaded from default location")
        except:
            try:
                ModelTrainer.trainer_y2()
                model_y2 = Utility.load('./data/model/model_y2.pkl')
            except:
                logging.info("[model_loader.py] Somewhere the problem occurs, please check!!!")
            logging.info("[model_loader.py] Model is not present is the default location. So, we're going to develop a model")
        return model_y2

    