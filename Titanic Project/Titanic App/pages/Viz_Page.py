import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import Titanic_Data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


test_merged = pd.merge(Titanic_Data.titanic_test, Titanic_Data.titanic_submission, how="inner")
titanic_data = pd.concat([Titanic_Data.titanic_train, test_merged], ignore_index=True)

df = titanic_data

st.title('Titanic Dataset Visualization')


st.header('Survival Rate')
survived_counts = df['Survived'].value_counts()
fig, ax = plt.subplots()
ax.bar(survived_counts.index.map({0: 'Not Survived', 1: 'Survived'}), survived_counts.values)
ax.set_ylabel('Count')
ax.set_title('Survival Count')
st.pyplot(fig)

st.title('Titanic Dataset Visualization with Interactive Charts')

st.header('Interactive Survival Rate')

survived_counts = df['Survived'].value_counts().reset_index()
survived_counts.columns = ['Survived', 'Count']
survived_counts['Survived'] = survived_counts['Survived'].map({0: 'Not Survived', 1: 'Survived'})

fig = px.pie(survived_counts, values='Count', names='Survived', title='Survival Rate', hover_data=['Count'], color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')

st.plotly_chart(fig, use_container_width=True)

st.header('Survivability by Gender')

survival_gender = df.groupby(['Sex', 'Survived'])['PassengerId'].count().unstack()
survival_gender['Survival Rate'] = survival_gender[1] / (survival_gender[0] + survival_gender[1]) * 100

survival_gender.reset_index(inplace=True)

fig = px.bar(survival_gender, 
             x='Sex', 
             y='Survival Rate', 
             text='Survival Rate',
             color='Sex',
             labels={'Survival Rate':'Survival Rate (%)'},
             title="Survival Rate by Gender")

fig.update_traces(texttemplate='%{text:.2s}%', textposition='outside')

fig.update_layout(xaxis_title="Gender",
                  yaxis_title="Survival Rate (%)",
                  uniformtext_minsize=8, 
                  uniformtext_mode='hide',
                  coloraxis_showscale=False)

st.plotly_chart(fig, use_container_width=True)

st.header('Survivability by Passenger Class')

survival_rates = df.groupby('Pclass')['Survived'].mean().reset_index()

fig = px.bar(survival_rates, x='Pclass', y='Survived', text='Survived',
             labels={'Survived': 'Survival Rate', 'Pclass': 'Passenger Class'},
             title='Survival Rates by Passenger Class')

fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

st.plotly_chart(fig, use_container_width=True)

st.header('Passenger Embarking Points')

locations = {
    'S': {'name': 'Southampton', 'lat': 50.9097, 'lon': -1.4044, 'color': 'blue'},
    'C': {'name': 'Cherbourg', 'lat': 49.6333, 'lon': -1.6164, 'color': 'green'},
    'Q': {'name': 'Queenstown', 'lat': 51.8500, 'lon': -8.2944, 'color': 'red'}
}

embark_counts = df['Embarked'].value_counts().reset_index()
embark_counts.columns = ['Embarked', 'Count']

for embark_code in locations:
    location = locations[embark_code]
    count = embark_counts.loc[embark_counts['Embarked'] == embark_code, 'Count'].values[0]
    location['count'] = count

fig = go.Figure()

for embark_code, info in locations.items():
    fig.add_trace(go.Scattergeo(
        lon = [info['lon']],
        lat = [info['lat']],
        text = f"{info['name']}<br>Passengers: {info['count']}",
        marker = dict(size = info['count']/10, color = info['color'], line_color='rgb(40,40,40)', line_width=0.5, sizemode = 'area'),
        name = f"{info['name']}"
    ))

fig.update_layout(
    title_text = 'Titanic Passengers Embarkation Points',
    geo=dict(
        scope='europe',
        landcolor='rgb(217, 217, 217)',
    )
)

st.plotly_chart(fig)

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

    plt.legend(loc="lower right")
    return plt

pipeline = Pipeline([
    ('scaler', MaxAbsScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=2))
])

X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived', 'Cabin', 'Embarked'])
y = df['Survived']

st.header('Learning Curve Graph')

plt = plot_learning_curve(pipeline, X, y)
st.pyplot(plt)

cv_scores = cross_val_score(pipeline, X, y, cv=10)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

st.write(f"Accuracy: {cv_mean:.2f} (+/- {cv_std * 2:.2f})")

# st.write("""---""")
# st.subheader("Here is my cross validation report")
# cv_mean, cv_std
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

# st.write("""---""")
# st.subheader("Here is a couple of different bar charts showing the number of passengers by class")
# st.write("""
# #### This one is interactive         
#          """)
# class_distribution = Titanic_Data.titanic_train['Pclass'].value_counts()
# st.bar_chart(class_distribution)
# st.write("""
# #### This one is just a picture         
#          """)
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Pclass', data=Titanic_Data.titanic_train)
# plt.title('Passenger Class Distribution')
# st.pyplot(plt)

# df1 = Titanic_Data.titanic_train
# passengers = df1[df1['PassengerId'].notna()]
# passengers['Survived'] = passengers['Survived'].astype(int)
# survival_by_gender = passengers.groupby('Sex')['Survived'].mean()
# survival_by_gender.index = ['Male', 'Female']
# st.subheader('Survival Rate By Gender')
# st.bar_chart(survival_by_gender)



