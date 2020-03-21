"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template,request
from FlaskWebProject1 import app
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Importing the dataset
def machine_model(name1):
    dataset = pd.read_csv(name1)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    total=cm[0][0]+[0][0]
    wrong=cm[0][1]+cm[1][0]
    return (((total-wrong)/total)*100)

def machine_model1(name1):
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting SVM to the Training set
    
    classifier = SVC(kernel='poly',random_state=0)
    classifier.fit(X_train,y_train)
    # Create your classifier here

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    total=cm[0][0]+[0][0]
    wrong=cm[0][1]+cm[1][0]
    return (((total-wrong)/total)*100)


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact', methods = ['POST'])  
def contact():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        name1=f.filename
        result=machine_model(name1)
        result1=machine_model1(name1)
        return render_template('contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.', name = f.filename,results=result,results1=result1)  

