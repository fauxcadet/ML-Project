import os 
import sys
import pickle
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split

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