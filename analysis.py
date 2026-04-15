import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

print("===== Data Overview =====")
print(data.info())
print("\n===== Describe Data =====")
print(data.describe(include='all'))

# Step 2: Missing Values
print("\n===== Missing Values Summary =====")
missing_summary = data.isnull().sum()
print(missing_summary[missing_summary > 0])

# Step 3: Data Cleaning
data.dropna(subset=['Age', 'Embarked'], inplace=True)
data['Cabin'].fillna('Unknown', inplace=True)

# Step 4: Univariate Analysis - Numerical Columns
num_columns = ['Age', 'Fare', 'SibSp', 'Parch']

for col in num_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    print(f"\nSummary of {col}")
    print(data[col].describe())

# Step 5: Univariate Analysis - Categorical Columns
cat_columns = ['Survived', 'Pclass', 'Sex', 'Embarked']

for col in cat_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=data[col])
    plt.title(f'Count Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

    print(f"\nValue Counts of {col}")
    print(data[col].value_counts())

# Step 6: Save Summary Report
summary_report = data.describe(include='all')
summary_report.to_csv('univariate_analysis_summary.csv', index=True)

print("\n✅ Univariate analysis summary saved as 'univariate_analysis_summary.csv'")