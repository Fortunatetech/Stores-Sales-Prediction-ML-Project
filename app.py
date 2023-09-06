from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Item_Weight = request.form.get("Item_Weight"),
            Item_Fat_Content = request.form.get('Item_Fat_Content'),
            Item_Type = request.form.get('Item_Type'),
            Item_MRP = request.form.get('Item_MRP'),
            Outlet_Size = request.form.get('Outlet_Size'),
            Outlet_Location_Type = request.form.get('Outlet_Location_Type'),
            Outlet_Type = request.form.get('Outlet_Type'),
            Outlet_Age = request.form.get('Outlet_Age')
            
            )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        # Round up the result to 2 decimal places and include the US dollar symbol
        return render_template('home.html', results=f'Prediction based on your inputs is ${results[0]:.2f}')

    

if __name__=="__main__":
    app.run(debug= True)        
    

