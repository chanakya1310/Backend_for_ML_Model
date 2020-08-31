from fastapi import FastAPI 
from typing import List
from pydantic import BaseModel
import numpy as np
import requests
import pickle

app = FastAPI()
filename = 'iris.pkl'
preprocess_fn = 'iris_preprocessing.pkl'
ss = pickle.load(open(preprocess_fn, 'rb'))
loaded_model = pickle.load(open(filename, 'rb'))

class Flower(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length: float
    petal_width: float

@app.get('/iris/test')
async def testing():
    return "Hello World"

@app.post('/iris/upload_one_flower/')
async def read_flower(flower : Flower):
    result = {}
    flower_to_predict = np.array([flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]).reshape(1, -1)
    flower_to_predict = ss.transform(flower_to_predict)
    if len(flower_to_predict[0]) == 4:
        prediction = loaded_model.predict(flower_to_predict)
        prediction = float(prediction[0])
        result.update({'prediction' : int(prediction)})
        if prediction == 1:
            result.update({'Flower name' : 'setosa'})
        elif prediction == 2:
            result.update({'Flower name' : 'versicolor'})
        elif prediction == 3:
            result.update({'Flower name' : 'virginica'})
    return result

@app.post('/iris/upload_multiple_flowers/')
async def read_flowers(flowers : List[Flower]):
    results = {}
    for i in range(0, len(flowers)):
        result = {}
        flower = flowers[i]
        flower_to_predict = np.array([flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]).reshape(1, -1)
        flower_to_predict = ss.transform(flower_to_predict)

        if len(flower_to_predict[0]) == 4:
            prediction = loaded_model.predict(flower_to_predict)
            prediction = float(prediction[0])
            result.update({'predicted class' : int(prediction)})
            if prediction == 0:
                result.update({'Predicted Flower Name' : 'setosa'})
            elif prediction == 1:
                result.update({'Predicted Flower Name' : 'versicolor'})
            elif prediction == 2:
                result.update({'Predicted Flower Name' : 'virginica'})
            

        results.update({'Flower ' + str(i + 1) : result})
        
    return results

