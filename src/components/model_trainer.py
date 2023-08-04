# The Packages/Methods which are necessary for Model training phase are imported here.
import os
import pandas as pd
import numpy as np
from src.logger import logging 
from src.exceptions import CustomException 
from src.components.data_transformation import DataTransformation
from catboost import CatBoostRegressor
from src.utils import Utility

# All operations related to Model Training phase are carried out inside the class 'ModelTrainer'
# We already know that, CatBoost Regressor performs well. Here, we're going to develop 2 models for Y1 and Y2 Prediction.
class ModelTrainer:
    def create_dir():
        Utility.create_directory('./data/model')

    def trainer_y1():
        ModelTrainer.create_dir()
        data = DataTransformation.final()
        temp_y1 = pd.DataFrame(data)
        # Creating model for prediction of Y1
        X, Y = temp_y1.drop(['Y1','Y2'],axis=1), temp_y1['Y1']
        model_y1 = CatBoostRegressor(verbose=0, n_estimators=10000,early_stopping_rounds=100).fit(X,Y)
        Utility.save(model_y1 , './data/model/model_y1.pkl')
        logging.info("[model_trainer.py] Models for Y1 are Saved Successfully")

    def trainer_y2():
        data = DataTransformation.final()
        temp_y2 = pd.DataFrame(data)
        # Creating model for prediction of Y2
        X, Y = temp_y2.drop(['Y2'],axis=1), temp_y2['Y2']
        model_y2 = CatBoostRegressor(verbose=0, n_estimators=10000,early_stopping_rounds=100).fit(X,Y)
        Utility.save(model_y2 , './data/model/model_y2.pkl')
        logging.info("[model_trainer.py] Models for Y2 are Saved Successfully")

    


