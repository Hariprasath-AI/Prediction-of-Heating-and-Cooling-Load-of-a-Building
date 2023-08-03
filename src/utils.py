# The methods which are used repeatedly and basic methods like loading, saving models are writtened in this utils.py file.
# The packages needs to imported for this utils file are listed below.
import os
from src.logger import logging
import pandas as pd
import pickle

# All the methods are defined inside the class 'Utility'
class Utility:
    # The 'create_directory' function gets the path as input parameter and creates a directory on that path.
    def create_directory(path):
        try:
            os.makedirs(path, exist_ok=True)
        except:
            logging.info("There was an error while creating a directory or 'Directory already exists'")

    # The 'save' function gets the path and model variable as input parameter. Then, it saves the model in pickle format in the path.
    def save(model, path):
        with open(path, 'wb') as file_obj:
            pickle.dump(model, file_obj)
            logging.info(f"Model saved in the {path} directory successfully")

    # The function 'load', gets the path as input parameter and load the model in that particular path.
    def load(path):
        with open(path, 'rb') as file_obj:
            model = pickle.load(file_obj)
            logging.info(f"Model Loaded in the {path} directory successfully")
        return model