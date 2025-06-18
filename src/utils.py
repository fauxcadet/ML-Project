import os 
import sys
import pickle
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves an object to a file using pickle.
    
    Parameters:
    file_path (str): The path where the object will be saved.
    obj: The object to be saved.
    
    Returns:
    None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            gs=GridSearchCV(model,para,cv=3,verbose=2,n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            logging.info(f"Best parameters for {list(models.keys())[i]}: {gs.best_params_}")
            model.fit(X_train, y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            logging.info(f"{list(models.keys())[i]}: {test_model_score}")
        return report
    except Exception as e:
        raise CustomException(e, sys) from e
def load_object(file_path):
    """
    Loads an object from a file using pickle.
    
    Parameters:
    file_path (str): The path from which the object will be loaded.
    
    Returns:
    obj: The loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e        