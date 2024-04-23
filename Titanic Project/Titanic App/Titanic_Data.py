import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

st.write("""
# Titanic Dataset Project         
         """)
st.write("""---""")
st.subheader('Summary')
st.write("""
#### This application is taking the passenger data from the famous sinking of the Titanic and displaying various graphs and charts. I have created some filters on the "Filters Page" along with some interactive charts on the "Viz Page".
         """)

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_submission = pd.read_csv('submission.csv')

titanic_train['Age'].fillna(value = round(titanic_train['Age'].mean()), inplace=True)

titanic_train['Sex'] = titanic_train.Sex.map({"male": 0, "female": 1})

X = titanic_train.drop(columns = ['PassengerId','Name','Ticket','Survived', 'Cabin', 'Embarked'],axis=1)
Y = titanic_train['Survived']

MaxAbsScaler = MaxAbsScaler()
X_Max = MaxAbsScaler.fit_transform(X)

KNN4 = KNeighborsClassifier(n_neighbors=2).fit(X_Max,Y)

print(round(KNN4.score(X_Max,Y), 4))

titanic_test['Age'].fillna(value = round(titanic_test['Age'].mean()), inplace=True)
titanic_test['Fare'].fillna(value = titanic_test['Fare'].mean(), inplace=True)
titanic_test['Sex'] = titanic_test.Sex.map({"male": 0, "female": 1})
X_test = titanic_test.drop(columns = ['PassengerId','Name','Ticket', 'Cabin', 'Embarked'],axis=1)
Y_test = titanic_submission['Survived']

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

Test_MaxAbsScaler = MaxAbsScaler()
Test_Max = Test_MaxAbsScaler.fit_transform(X_test)

KNN4 = KNeighborsClassifier(n_neighbors=2).fit(Test_Max,Y_test)

print("KNN 4")
print(round(KNN4.score(Test_Max,Y_test), 4))

pipeline = Pipeline([
    ('scaler', MaxAbsScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=2))
])

def plot_learning_curve(estimator, X, y, title='Learning Curve'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color="r", alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color="g", alpha=0.1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

cv_scores = cross_val_score(pipeline, X, Y, cv=10)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

st.write(f"Accuracy: {cv_mean:.2f} (+/- {cv_std * 2:.2f})")

