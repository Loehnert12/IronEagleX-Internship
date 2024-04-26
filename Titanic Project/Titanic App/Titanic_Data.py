import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import streamlit as st

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
titanic_submission = pd.read_csv('submission.csv')

imputer = SimpleImputer(strategy='mean')
titanic_train['Age'] = imputer.fit_transform(titanic_train[['Age']])
titanic_train['Sex'] = titanic_train['Sex'].map({"male": 0, "female": 1})
titanic_test['Age'] = imputer.transform(titanic_test[['Age']])
titanic_test['Fare'].fillna(titanic_test['Fare'].mean(), inplace=True)
titanic_test['Sex'] = titanic_test['Sex'].map({"male": 0, "female": 1})

X_train = titanic_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived', 'Cabin', 'Embarked'])
Y_train = titanic_train['Survived']
X_test = titanic_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
Y_test = titanic_submission['Survived']

pipeline = Pipeline([
    ('scaler', MaxAbsScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=2))
])
pipeline.fit(X_train, Y_train)

# Evaluating on the training data
print(f"Training Accuracy: {round(pipeline.score(X_train, Y_train), 4)}")

# Cross-validation to check robustness
cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=10)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"Cross-Validation Score: {cv_mean:.4f} ± {cv_std:.4f}")

st.title('Titanic Survival Prediction Project')
st.write('''
## Summary
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

### **What Data Will I Use?**

I will use two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled train.csv and the other is titled test.csv.

Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.
''')

st.header('Sample Data')
st.write("""This is the data that I will train with my model after I cleaned it and filled in missing values""")
st.write(titanic_train.head())
st.write("""And here is the data after cleaning it and removing columns I do not need""")
st.write(X_train.head())
st.subheader('Model Used: K-Nearest Neighbors (KNN)')
st.write('KNN is a supervised machine learning algorithm often used for classification tasks. It works by understanding how data points are grouped together (based on features) and then using this knowledge to classify new, unseen data points.')
st.subheader('Model Details')
st.write('This KNN model was the best one that I found after running the data through multiple other models using a MaxAbsScaler. The training accuracy for this model is currently 89%')