import pickle
from flask import Flask,request,render_template,jsonify,url_for,app
import numpy as np
import pandas as pd

app=Flask(__name__)
scaler=pickle.load(open('scaler.pkl','rb'))
heart_model=pickle.load(open('heart_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.form
    print(data)
    data=np.array(list(data.values()),dtype=float).reshape(1,-1)
    new_data=scaler.transform(data)
    output=heart_model.predict(new_data)
    probability = heart_model.predict_proba(new_data)[0][1] * 100
    print(output[0])
    if output[0]==1:
        prediction_text='The person is likely to have heart disease.'
        precautions = [
        "Consult a cardiologist immediately",
        "Reduce oily and junk foods",
        "Avoid smoking and alcohol",
        "Do regular walking and light exercise",
        "Monitor blood pressure regularly",
        "Reduce stress and get proper sleep"
    ]
    else:
        prediction_text='The person is unlikely to have heart disease.'
        precautions = [
        "Maintain a healthy diet",
        "Exercise regularly",
        "Avoid smoking and excessive alcohol consumption",
        "Manage stress levels",
        "Get regular health check-ups"
    ]
    return render_template('home.html',prediction_text=prediction_text,precautions=precautions,probability=probability)

if __name__=="__main__":
    app.run(host="0.0.0.0")