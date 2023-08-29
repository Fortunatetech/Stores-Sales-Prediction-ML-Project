import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
#from src.components.data_ingestion import DataIngestion

@dataclass
class DataCleaningConfig:
    train_data_path_cleaned: str=os.path.join('artifacts',"train_cleaned.csv")
    test_data_path_cleaned: str=os.path.join('artifacts',"test_cleaned.csv")

class DataCleaning:
    def __init__(self):
        self.cleaning_config=DataCleaningConfig()
    
    def initiate_data_cleaning(self):
        logging.info("Entered the data cleaning method or component")
        try:
            df_train=pd.read_csv("artifacts/train.csv")
            df_test=pd.read_csv("artifacts/test.csv")
        
            logging.info('Read the dataset as dataframe')
            #creating our new column for both datasets
            df_train['Outlet_Age']= df_train['Outlet_Establishment_Year'].apply(lambda year: 2023 - year).astype(int)
            df_test['Outlet_Age']= df_test['Outlet_Establishment_Year'].apply(lambda year: 2023 - year).astype(int)
            # Standardize values in the 'Item_Fat_Content' column
            df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
            df_train.drop(['Item_Identifier','Outlet_Identifier', 'Item_Visibility', 'Outlet_Establishment_Year'], axis = 1, inplace  = True)

            df_test['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
            df_test.drop(['Item_Identifier','Outlet_Identifier', 'Item_Visibility', 'Outlet_Establishment_Year'], axis = 1, inplace  = True)
           
            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path_cleaned),exist_ok=True)
            logging.info('directory created')


            df_train.to_csv(self.cleaning_config.train_data_path_cleaned,index=False,header=True)
            logging.info('Train data saved')
            df_test.to_csv(self.cleaning_config.test_data_path_cleaned,index=False,header=True)
            logging.info('Test data saved')

            logging.info("Train and test data cleaned")
            return df_train, df_test
            logging.info("returned df_train and df_test")
            
        
        except Exception as e:
            raise CustomException(e,sys)
    





            

            

        
           


