import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_submission = pd.read_csv('submission.csv')
titanic_train['Age'].fillna(value=round(titanic_train['Age'].mean()), inplace=True)
titanic_train['Sex'] = titanic_train.Sex.map({"male": 0, "female": 1})
X = titanic_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived', 'Cabin', 'Embarked'], axis=1)
Y = titanic_train['Survived']

pipeline = Pipeline([
    ('scaler', MaxAbsScaler()),  # Preprocessing: Scale data using MaxAbsScaler instance
    ('knn', KNeighborsClassifier(n_neighbors=2))  # Classifier: KNN
])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

pipeline.fit(X_train, Y_train)

cv_scores = cross_val_score(pipeline, X, Y, cv=10)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# cv_mean, cv_std

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, Y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.title('Learning Curve')
plt.xlabel('Training Sizes')
plt.ylabel('Score')
plt.legend()
plt.show()

st.pyplot(plt.gcf())

st.write("""
# Titanic Dataset Project         
         """)
st.write("""---""")
st.subheader('Summary')
st.write("""
#### This application is taking the passenger data from the famous sinking of the Titanic and displaying various graphs and charts. I have created some filters on the "Filters Page" along with some interactive charts on the "Viz Page".
         """)