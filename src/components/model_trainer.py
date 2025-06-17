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
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']     
                                  },
                "RandomForestRegressor": {
                    'n_estimators':[8,16,31,64,128,256],
                },
                "GradientBoostingRegressor": {
                    'n_estimators':[8,16,31,64,128,256],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'max_depth':[3,5,7,9]
                },
                "LinearRegression": {
                },
                "KNeighborsRegressor": {
                    'n_neighbors':[3,5,7,9,11,13],
                  #  'weights':['uniform','distance'],
                   # 'algorithm':['auto','ball_tree','kd_tree','brute']
                },
                "XGBRegressor": {
                    'learning_rate':[0.01,0.05,0.1,0.2],
                    'n_estimators':[8,16,31,64,128,256],
                },
                "CatBoostRegressor": {
                    'learning_rate':[0.01,0.05,0.1,0.2],
                    #'n_estimators':[8,16,31,64,128,256],
                    'depth':[3,5,7,9,6,8,10],
                    'iterations':[30,50,100,200,300,400,500],
                },
                "AdaBoostRegressor": {
                    'n_estimators':[8,16,31,64,128,256],
                    'learning_rate':[0.001,0.01,0.05,0.1,0.2]
                }
                
            }

            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params 
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

