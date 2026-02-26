# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Start by Importing the Necessary libraries. 
2)Import and Load the Data. 
3)Split Dataset into Training and Testing Sets. 
4)Train the Model Using Stochastic Gradient Descent (SGD). 
5)Make Predictions and Evaluate Accuracy. 
6)Generate Confusion Matrix. 
7)Stop. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Jegan S K
RegisterNumber:  212225230117
*/

# import all the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris

# X & y data
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']

# Training and Testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=SGDClassifier(max_iter=1000,tol=1e-3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Performance measures
acc=accuracy_score(y_test,y_pred)
con=confusion_matrix(y_test,y_pred)

#displaying all the outputs
print("\nY_pred=")
print(y_pred)
print("\nAccuracy Score : ",acc)
print("\nConfusion Matrix :")
print(con)

```

## Output:

<img width="819" height="497" alt="image" src="https://github.com/user-attachments/assets/2e81b188-3921-48bb-9852-1b4e45cdb760" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
