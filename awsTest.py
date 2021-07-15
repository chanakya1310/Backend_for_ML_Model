from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import boto3
import numpy as np
import pickle
from io import BytesIO
from boto.s3.key import Key
from boto.s3.connection import S3Connection


BUCKET_NAME = 'iris-test1310'
MODEL_FILE_NAME = 'iris.pkl'
MODEL_LOCAL_PATH = 'iris.pkl'

app = FastAPI()
# filename = 'iris.pkl'
# preprocess_fn = 'iris_preprocessing.pkl'
# ss = pickle.load(open(preprocess_fn, 'rb'))

class Flower(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length: float
    petal_width: float

@app.get('/iris/test')
async def testing():
    return "Hello World"

@app.get('/iris/createbucket')
async def createBucket():
    s3 = boto3.resource('s3')
    try:
        bucket = s3.create_bucket(Bucket = BUCKET_NAME, CreateBucketConfiguration={
        'LocationConstraint': 'ap-south-1'})
        s3.Object(BUCKET_NAME, 'iris.pkl').put(Body=open('iris.pkl', 'rb'))
        s3.Object(BUCKET_NAME, 'iris_preprocessing.pkl').put(Body=open('iris_preprocessing.pkl', 'rb'))
        return {"Bucket Successfully Created"}
    except Exception as e:
        return {'S3 error: ', e}

@app.post('/iris/upload_one_flower/')
async def read_flower(flower : Flower):
    result = {}
    flower_to_predict = np.array([flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]).reshape(1, -1)
    if len(flower_to_predict[0]) == 4:
        prediction = predict(flower_to_predict)
        prediction = float(prediction[0])
        result.update({'prediction' : int(prediction)})
        if prediction == 0:
            result.update({'Flower name' : 'setosa'})
        elif prediction == 1:
            result.update({'Flower name' : 'versicolor'})
        elif prediction == 2:
            result.update({'Flower name' : 'virginica'})
    return result

def load_model():
    s3 = boto3.resource('s3')

    with BytesIO() as data:
        s3.Bucket(BUCKET_NAME).download_fileobj("iris_preprocessing.pkl", data)
        data.seek(0)    # move back to the beginning after writing
        ss = pickle.load(data)

    with BytesIO() as data:
        s3.Bucket(BUCKET_NAME).download_fileobj("iris.pkl", data)
        data.seek(0)    # move back to the beginning after writing
        model = pickle.load(data)

    return [ss, model]

def predict(data):
  ss, model = load_model()
#   print("Before transform", data)
  data = ss.transform(data)
#   print("After transform", data)
  prediction = model.predict(data)
  print(prediction)
  return prediction