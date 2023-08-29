import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        Item_Weight: float,
        Item_Fat_Content: str,
        Item_Type: str,
        Item_MRP : float,
        Outlet_Size: str,
        Outlet_Location_Type: str,
        Outlet_Type: str,
        Outlet_Age: int
        ):

        self.Item_Weight = Item_Weight

        self.Item_Fat_Content = Item_Fat_Content

        self.Item_Type = Item_Type

        self.Item_MRP = Item_MRP

        self. Outlet_Size =  Outlet_Size

        self.Outlet_Location_Type = Outlet_Location_Type

        self.Outlet_Type = Outlet_Type

        self.Outlet_Age = Outlet_Age
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Item_Weight": [self.Item_Weight],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Type": [self.Item_Type],
                "Item_MRP": [self.Item_MRP],
                "Outlet_Size": [self.Outlet_Size],
                "Outlet_Location_Type": [self.Outlet_Location_Type],
                "Outlet_Type": [self.Outlet_Type],
                "Outlet_Age": [self.Outlet_Age],                            
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)