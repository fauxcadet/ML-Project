import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This method creates a preprocessor object that applies transformations to the dataset.
        It includes pipelines for numerical and categorical features, handling missing values,
        scaling numerical features, and encoding categorical features.
        """
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns =["gender","race/ethnicity","parental level of education",
                                  "lunch","test preparation course"]
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Fill missing values with the median
                ('scaler', StandardScaler()) # Scale numerical features
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),# Fill missing values with the most frequent value
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),# Encode categorical features
                ('scaler', StandardScaler(with_mean=False))# Scale categorical features
            ])
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns), # Apply numerical pipeline to numerical columns
                    ('cat_pipeline', cat_pipeline, categorical_columns) # Apply categorical pipeline to categorical columns
                ]
            )
            logging.info("Column transformer object created")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
    def initiate_data_transformation(self, train_path, test_path):
        """
        This method initiates the data transformation process.
        It reads the training and testing datasets, applies the preprocessor,
        and saves the preprocessor object to a file.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed Under Data Transformation")
            logging.info("Obtaining preprocessing Object")
            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math score' # Specify the target column name
            numerical_columns = ['writing_score', 'reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
           # print("Input feature train columns:", input_feature_train_df.columns.tolist())
           # print("Input feature test columns:", input_feature_test_df.columns.tolist())

            logging.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) #` Fit and transform the training data`
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)# ` Transform the testing data`

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # Combine input features and target variable for training data
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]# Combine input features and target variable for testing data

            logging.info("Preprocessing completed")
            input_feature_test_arr= preprocessing_obj.fit_transform(input_feature_test_df) # Fit the preprocessor on the training data
            input_feature_train_arr= preprocessing_obj.transform(input_feature_train_df) # Fit the preprocessor on the training data

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # Combine input features and target variable for training data
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # Combine input features and target variable for testing data
            logging.info("Saved preprocessing object")
            # Save the preprocessor object to a file

            save_object (
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys) from e
"""f __name__ == "__main__":
    obj = DataTransformation()
    train_arr, test_arr, preprocessor_file_path = obj.initiate_data_transformation(
        train_path='artifact/train.csv',
        test_path='artifact/test.csv'
    )
    print("Train Array:", train_arr)
    print("Test Array:", test_arr)
    print("Preprocessor File Path:", preprocessor_file_path)
# This code is for the data transformation component of a machine learning project.    """