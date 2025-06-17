import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import ( RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor,)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.utils import r2_score
from dataclasses import dataclass
from src.utils import r2_score
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'XGBRegressor': XGBRegressor(objective='reg:squarederror'),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor()
            }

            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            logging.info("Evaluating models")
            ## to get best model score from model report
            best_model_score = max(sorted(model_report.values()))
            ## to choose best model name from model report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score", sys)
            logging.info(f"Saving best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys) from e    

