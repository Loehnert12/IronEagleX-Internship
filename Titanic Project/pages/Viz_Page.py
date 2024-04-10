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



