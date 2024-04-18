import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
import plotly.express as px 
import plotly.graph_objects as go

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_submission = pd.read_csv('submission.csv')
titanic_train.head()

titanic_train['Age'].fillna(value = round(titanic_train['Age'].mean()), inplace=True)

titanic_train['Sex'] = titanic_train.Sex.map({"male": 0, "female": 1})

X = titanic_train.drop(columns = ['PassengerId','Name','Ticket','Survived', 'Cabin', 'Embarked'],axis=1)
Y = titanic_train['Survived']

MinMaxScaler = MinMaxScaler()
X_Min = MinMaxScaler.fit_transform(X)

RobustScaler = RobustScaler()
X_Robust = RobustScaler.fit_transform(X)

MaxAbsScaler = MaxAbsScaler()
X_Max = MaxAbsScaler.fit_transform(X)

KNN1 = KNeighborsClassifier(n_neighbors=2).fit(X,Y)
KNN2 = KNeighborsClassifier(n_neighbors=2).fit(X_Min,Y)
KNN3 = KNeighborsClassifier(n_neighbors=2).fit(X_Robust,Y)
KNN4 = KNeighborsClassifier(n_neighbors=2).fit(X_Max,Y)

print(round(KNN1.score(X,Y), 4))
print(round(KNN2.score(X_Min,Y), 4))
print(round(KNN3.score(X_Robust,Y), 4))
print(round(KNN4.score(X_Max,Y), 4))

titanic_test['Age'].fillna(value = round(titanic_test['Age'].mean()), inplace=True)
titanic_test['Fare'].fillna(value = titanic_test['Fare'].mean(), inplace=True)
titanic_test['Sex'] = titanic_test.Sex.map({"male": 0, "female": 1})
X_test = titanic_test.drop(columns = ['PassengerId','Name','Ticket', 'Cabin', 'Embarked'],axis=1)
Y_test = titanic_submission['Survived']

titanic_test.describe()

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

Test_MinMaxScaler = MinMaxScaler()
Test_Min = Test_MinMaxScaler.fit_transform(X_test)

Test_RobustScaler = RobustScaler()
Test_Robust = Test_RobustScaler.fit_transform(X_test)

Test_MaxAbsScaler = MaxAbsScaler()
Test_Max = Test_MaxAbsScaler.fit_transform(X_test)

KNN1 = KNeighborsClassifier(n_neighbors=2).fit(X_test,Y_test)
KNN2 = KNeighborsClassifier(n_neighbors=2).fit(Test_Min,Y_test)
KNN3 = KNeighborsClassifier(n_neighbors=2).fit(Test_Robust,Y_test)
KNN4 = KNeighborsClassifier(n_neighbors=2).fit(Test_Max,Y_test)

print(round(KNN1.score(X,Y), 4))
print(round(KNN2.score(X_Min,Y), 4))
print(round(KNN3.score(X_Robust,Y), 4))
print(round(KNN4.score(X_Max,Y), 4))

print(round(KNN1.score(X_test,Y_test), 4))
print(round(KNN2.score(Test_Min,Y_test), 4))
print(round(KNN3.score(Test_Robust,Y_test), 4))
print(round(KNN4.score(Test_Max,Y_test), 4))

st.write("""
# Titanic Dataset Project         
         """)
st.write("""---""")
st.subheader('Summary')
st.write("""
#### This application is taking the passenger data from the famous sinking of the Titanic and displaying various graphs and charts. I have created some filters on the "Filters Page" along with some interactive charts on the "Viz Page".
         """)