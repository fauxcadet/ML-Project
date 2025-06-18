import sys
import pandas as pd 
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        """
        This method predicts the target variable using the preprocessor and model.
        It loads the preprocessor and model objects, transforms the input features,
        and returns the predicted value.
        """
        try:

            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model= load_object(file_path=model_path)
            preprocessor= load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
            #print("🔎 Input to model:\n", features)
            #print("📊 Scaled Data:\n", data_scaled)

            return preds 
        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education: str,
        lunch:str ,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int

        ):

        self.gender=gender
        self.race_ethnicity= race_ethnicity
        self.parental_level_of_education= parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score      
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {  
                "gender": [self.gender],
                "race/ethnicity":[self.race_ethnicity] ,
                "parental level of education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test preparation course":[self.test_preparation_course],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score]
            } 
            df=  pd.DataFrame(custom_data_input_dict)
           # print("Custom Data Frame:\n", df)
            return df
            #return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e

