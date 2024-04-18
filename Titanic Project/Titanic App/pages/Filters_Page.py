import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import Titanic_Data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

test_merged = pd.merge(Titanic_Data.titanic_test, Titanic_Data.titanic_submission, how="inner")
titanic_data = pd.concat([Titanic_Data.titanic_train, test_merged], ignore_index=True)

df = titanic_data

st.write("""
## I built a search filter to find specific passengers.         
         """)

min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age = st.slider('Age', min_age, max_age, (min_age, max_age))

gender = st.selectbox('Gender', ['Any', 'Male', 'Female'])

pclass_options = ['Any'] + sorted(df['Pclass'].unique().tolist())
pclass = st.selectbox('Passenger Class', pclass_options)

survival_status = st.selectbox('Survival Status', ['Any', 'Survived', 'Did Not Survive'])

search_query = st.text_input('Search by Name or Ticket Number')

filtered_df = df[(df['Age'].between(*age)) &
                 (df['Sex'] == gender if gender != 'Any' else True) &
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