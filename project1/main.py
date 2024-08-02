from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def main():
 return {"Machine Learning Model In Deployment"}

@app.get('/{name}')
def ml(name: str):
 return {'message' : f'Welcome to MLOPs {name}'}

from sklearn.datasets import load_iris
#from sklearn.navie_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB

#Load Iris Datasets

iris = load_iris()

X = iris.data
y = iris.target

#Fitting our model on the dataset

clf = GaussianNB()
clf.fit(X,y)

from pydantic import BaseModel

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

@app.post('/predict')
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}
 
