import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Step 1: Load the dataset
st.title('Univariate Analysis Project')
st.write('Exploratory analysis of the Titanic dataset using Streamlit.')

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Step 2: Data Overview
st.header('Data Overview')
st.write(data.info())
st.dataframe(data.describe(include='all'))

# Step 3: Data Cleaning
missing_summary = data.isnull().sum()
st.header('Missing Values Summary')
st.dataframe(missing_summary[missing_summary > 0])

data.dropna(subset=['Age', 'Embarked'], inplace=True)
data['Cabin'].fillna('Unknown', inplace=True)

# Step 4: Univariate Analysis - Numerical Columns
st.header('Univariate Analysis - Numerical Columns')
num_columns = ['Age', 'Fare', 'SibSp', 'Parch']
for col in num_columns:
    st.subheader(f'Distribution of {col}')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data[col], kde=True, bins=30, color='skyblue', ax=ax)
    st.pyplot(fig)
    st.write(data[col].describe())

# Step 5: Univariate Analysis - Categorical Columns
st.header('Univariate Analysis - Categorical Columns')
cat_columns = ['Survived', 'Pclass', 'Sex', 'Embarked']
for col in cat_columns:
    st.subheader(f'Count Distribution of {col}')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=data[col], palette='viridis', ax=ax)
    st.pyplot(fig)
    st.write(data[col].value_counts())

# Step 6: Summary Report
summary_report = data.describe(include='all')
summary_report.to_csv('univariate_analysis_summary.csv', index=True)
st.success('Univariate analysis summary saved as univariate_analysis_summary.csv')
