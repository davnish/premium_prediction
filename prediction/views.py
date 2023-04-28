from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics


data = pd.read_csv('/Users/nischalsingh/Programming/machine_learning/insurance.csv')

cat_features = data.dtypes[data.dtypes == 'object'].index

data = data.drop_duplicates()

dummies = pd.get_dummies(data.sex)
merge = pd.concat([data, dummies], axis = 1)
data = merge.drop(['sex','female'],axis = 1)
data['male'] = data['male'].astype(int) 

dummies = pd.get_dummies(data.smoker)
merge = pd.concat([data, dummies], axis = 1)
data = merge.drop(['smoker','no'],axis = 1)
data['yes'] = data['yes'].astype(int) 

dummies = pd.get_dummies(data.region)
merge = pd.concat([data, dummies], axis = 1)
data = merge.drop(['region'],axis = 1)
data = data.astype({'northeast':int, 'northwest':int, 'southeast': int, 'southwest': int}) 

target_name = 'expenses'
y = data[target_name]

X = data.drop(target_name, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 7)

lr = LinearRegression()
lr.fit(X_train.values, y_train.values)
y_pred = lr.predict(X_test.values)

score = r2_score(y_test, y_pred)
MAE= metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

@csrf_exempt
def modelprediction(request):
    region = {"00":[1,0,0,0],"01":[0,1,0,0], "10":[0,0,1,0], "11":[0,0,0,1]}
    predicted = lr.predict([[int(request.POST.get('age')), int(request.POST.get('bmi')), 
                            int(request.POST.get('children')), int(request.POST.get('gender')),
                            int(request.POST.get('smoker')),
                            *region[request.POST.get('region')]]])
    return HttpResponse(predicted)

def prediction(request):
    return render(request, 'prediction/base.html', {'r2_score': score, 'MAE':MAE, 'MSE':MSE, 'RMSE':RMSE})
