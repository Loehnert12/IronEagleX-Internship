import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import Titanic_Data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import TitanicV2 as titanic

test_merged = pd.merge(titanic.titanic_test, titanic.titanic_submission, how="inner")
titanic_data = pd.concat([titanic.titanic_train, test_merged], ignore_index=True)

df = titanic_data

st.write("""
# Welcome to my filters page!""")
st.write("""
#### Now we can utilize the model with some different filters that I created.          
         """)

min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age = st.slider('Age', min_age, max_age, (min_age, max_age))

sex_map = {'male': 0, 'female': 1}

gender = st.selectbox('Gender', ['Any', 'Male', 'Female'])

pclass_options = ['Any'] + sorted(df['Pclass'].unique().tolist())
pclass = st.selectbox('Passenger Class', pclass_options)

survival_status = st.selectbox('Survival Status', ['Any', 'Survived', 'Did Not Survive'])

search_query = st.text_input('Search by Name or Ticket Number')

reverse_sex_map = {v: k for k, v in sex_map.items()}

df['Sex'] = df['Sex'].map(reverse_sex_map)

filtered_df = df[(df['Age'].between(*age)) &
                 (df['Sex'] == gender.lower() if gender != 'Any' else True) & 
                 (df['Pclass'] == pclass if pclass != 'Any' else True) &
                 (df['Survived'] == (1 if survival_status == 'Survived' else 0) if survival_status != 'Any' else True) &
                 (df['Name'].str.contains(search_query, case=False) | df['Ticket'].str.contains(search_query, case=False))]

st.dataframe(filtered_df)

st.header('Age Distribution by Group')

st.write("""
#### Here is a filter to look at the counts by different groups         
         """)

group_type = st.selectbox('Select a group to filter by:', ['Passenger Class', 'Gender', 'Survival Status'])

if group_type == 'Passenger Class':
    group_column = 'Pclass'
elif group_type == 'Gender':
    group_column = 'Sex'
else:  
    group_column = 'Survived'

    df['Survived'] = df['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})

fig = px.histogram(df, x='Age', color=group_column, nbins=30,
                   title=f'Age Distribution by {group_type}',
                   labels={'Age': 'Age', group_column: group_type},
                   template='plotly_white', barmode='overlay')

fig.update_traces(opacity=0.75)
st.plotly_chart(fig, use_container_width=True)

st.title("Titanic Survival Prediction")
st.write("Please enter your information:")

# User input fields
age = st.slider("Age", 1, 100, 28)
sex = st.selectbox("Sex", options=['Male', 'Female'])
pclass = st.selectbox("Passenger Class", options=[1, 2, 3])
siblings_spouses_aboard = st.slider("Siblings/Spouses Aboard", 0, 10, 1)
parents_children_aboard = st.slider("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", value=20.0)

# Map user input to data format expected by model
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [0 if sex == 'Male' else 1],
    'Age': [age],
    'SibSp': [siblings_spouses_aboard],
    'Parch': [parents_children_aboard],
    'Fare': [fare]
})

if st.button("Predict Survival"):
    prediction = titanic.pipeline.predict(input_data)
    result = "Survive" if prediction[0] == 1 else "Not Survive"
    st.write(f"Prediction: You would likely {result}")