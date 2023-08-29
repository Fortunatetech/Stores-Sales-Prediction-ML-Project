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
            df1=pd.read_csv("artifacts/train.csv")
            df2=pd.read_csv("artifacts/test.csv")
        
            logging.info('Read the dataset as dataframe')


            df1['Journey_day']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.day
            df1['Journey_month']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.month
            df1['Journey_year']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.year
            df1= df1.drop(['Date_of_Journey'], axis=1)
            logging.info('processingthe training dataset as dataframe')
            df1['hours']=pd.to_datetime(df1['Dep_Time']).dt.hour
            df1['minutes']=pd.to_datetime(df1['Dep_Time']).dt.minute
            df1.drop(["Dep_Time"], axis = 1, inplace = True)
            df1["Arrival_hour"] = pd.to_datetime(df1.Arrival_Time).dt.hour
            df1["Arrival_min"] = pd.to_datetime(df1.Arrival_Time).dt.minute
            df1 = df1.drop(["Additional_Info"],axis=1)
            duration = list(df1["Duration"])

            for i in range(len(duration)):
                
                if len(duration[i].split()) != 2:
                      # Check if duration contains only hour or mins
                    if "h" in duration[i]:
                    
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        
                        duration[i] = "0h " + duration[i]  
                        
            duration_hours = []
            duration_mins = []
            for i in range(len(duration)):
                duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
            df1["duration_mins"]= duration_mins
            df1["duration_hours"]= duration_hours
            df1 = df1.drop(["Duration"],axis=1)
            df1 = df1.drop(["Arrival_Time"],axis=1) 
            df1 = df1.drop(["Route"],axis=1)
            df1.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


            df2['Journey_day']=pd.to_datetime(df2['Date_of_Journey'],format="%d/%m/%Y").dt.day
            df2['Journey_month']=pd.to_datetime(df2['Date_of_Journey'],format="%d/%m/%Y").dt.month
            df2['Journey_year']=pd.to_datetime(df2['Date_of_Journey'],format="%d/%m/%Y").dt.year
            df2= df2.drop(['Date_of_Journey'], axis=1)
            df2['hours']=pd.to_datetime(df2['Dep_Time']).dt.hour
            df2['minutes']=pd.to_datetime(df2['Dep_Time']).dt.minute
            df2.drop(["Dep_Time"], axis = 1, inplace = True)
            df2["Arrival_hour"] = pd.to_datetime(df2.Arrival_Time).dt.hour
            df2["Arrival_min"] = pd.to_datetime(df2.Arrival_Time).dt.minute
            df2 = df2.drop(["Additional_Info"],axis=1)
            logging.info('processingthe test dataset as dataframe')
            duration = list(df2["Duration"])

            for i in range(len(duration)):
                #logging.info('entered the loop')
                
                if len(duration[i].split()) != 2:
                      # Check if duration contains only hour or mins
                    if "h" in duration[i]:
                    
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        
                        duration[i] = "0h " + duration[i]  
            logging.info('ended the loop')
            duration_hours = []
            duration_mins = []
            for i in range(len(duration)):
                #logging.info('entered the loop')
                duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
            df2["duration_mins"]= duration_mins
            df2["duration_hours"]= duration_hours
            df2 = df2.drop(["Duration"],axis=1)
            df2 = df2.drop(["Arrival_Time"],axis=1)
            df2 = df2.drop(["Route"],axis=1)
            df2.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
            

            logging.info('ended he loop') 
        
            logging.info('returning done') 

            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path_cleaned),exist_ok=True)
            logging.info('directory created')


            df1.to_csv(self.cleaning_config.train_data_path_cleaned,index=False,header=True)
            logging.info('df1 saved')
            df2.to_csv(self.cleaning_config.test_data_path_cleaned,index=False,header=True)
            logging.info('df2 saved')

            logging.info("Train test data cleaned")
            return df1,df2 
            logging.info("returned df1 and df2")
            # Adds 0 hour

        
        except Exception as e:
            raise CustomException(e,sys)
    





            

            

        
           


