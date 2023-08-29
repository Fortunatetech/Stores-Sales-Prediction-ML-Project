import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class LabelEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.apply(LabelEncoder().fit_transform)
        return X_encoded

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            Label = ['Item_Fat_Content', 'Outlet_Location_Type']
            cols = ['Item_Type','Outlet_Size', 'Outlet_Type']
            col_to_scale = ['Item_Weight', 'Item_MRP', 'Outlet_Age']

            oh_cols = Pipeline(steps=[  
                ("imputer",SimpleImputer(strategy="most_frequent")),             
                ('one_hot_encoder', OneHotEncoder(drop='first')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            col_to_scale_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])

            logging.info(f"Features to label-encode: {Label}")
            logging.info(f"Features to one-hot-encode: {cols}")
            logging.info(f"Features to scale: {col_to_scale}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('label_encode', LabelEncodeTransformer(), Label),
                    ('oh_cols', oh_cols, cols),
                    ('col_to_scale', col_to_scale_pipeline, col_to_scale),
                ]
            )

            logging.info("Train test split initiated")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def inititate_data_transformation(self):
        try:
            train_df = pd.read_csv("artifacts/train_cleaned.csv")
            test_df = pd.read_csv("artifacts/test_cleaned.csv")         

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation()
            target_column_name = "Item_Outlet_Sales"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Saved preprocessing object.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (
                train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# Usage example
if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_file = data_transformation.inititate_data_transformation()
