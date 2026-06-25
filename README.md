 🚢 Titanic Dataset Analysis

 📖 Project Overview

This project performs Exploratory Data Analysis (EDA), Data Cleaning, Missing Value Treatment, and Feature Engineering on the famous Titanic dataset.

The goal is to understand the factors that influenced passenger survival and prepare the dataset for machine learning modeling.



 🎯 Objectives

- Clean and preprocess the Titanic dataset
- Handle missing values effectively
- Perform correlation analysis
- Convert categorical variables into numerical format
- Identify important factors affecting passenger survival
- Prepare the dataset for predictive modeling

---

 📂 Dataset

The dataset contains information about Titanic passengers, including:

- Passenger Class (`pclass`)
- Survival Status (`survived`)
- Gender (`sex`)
- Age (`age`)
- Ticket Fare (`fare`)
- Family Members (`sibsp`, `parch`)
- Embarkation Details
- Passenger Category (`who`)
- Cabin Deck Information

Dataset Source:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv



 🛠️ Data Cleaning

 1. Removed High Missing Value Column

The `deck` column contained a large number of missing values (>70%) and was removed.

```python
df_titanic.drop('deck', axis=1, inplace=True)
2. Removed Missing Embarkation Records

Only two records contained missing values in embark_town, so they were safely removed.

df_titanic.drop([61, 829], inplace=True)
🔍 Missing Value Treatment
Age Imputation

Missing age values were filled using the average age of each passenger class.

Passenger Class	Average Age
First Class	38
Second Class	29
Third Class	25
def impute_age(cols):
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    return age

This approach preserves class-specific age distributions better than using a global mean.

⚙️ Feature Engineering
Gender Encoding

The sex column was converted into numeric format using One-Hot Encoding.

gender = pd.get_dummies(df_titanic['sex'], drop_first=True)

Generated feature:

male
Passenger Type Encoding

The who column was encoded into:

child
man
woman
who = pd.get_dummies(df_titanic['who'])
Embarkation Town Encoding

The embark_town column was transformed into dummy variables.

embark_town = pd.get_dummies(df_titanic['embark_town'])
📊 Exploratory Data Analysis

A correlation matrix was generated to identify relationships between features and survival.

Key Findings
👨 Gender Had the Strongest Impact
Feature	Correlation with Survival
male	-0.542
woman	+0.504
adult_male	-0.556
Insight
Male passengers were significantly less likely to survive.
Female passengers had much higher survival rates.
Supports the historical "Women and Children First" evacuation policy.
🎟 Passenger Class Influenced Survival
Feature	Correlation
pclass	-0.336
Insight

Passenger classes:

1 = First Class
2 = Second Class
3 = Third Class

Higher class passengers had better survival chances.

First Class > Second Class > Third Class
💰 Ticket Fare Affected Survival
Feature	Correlation
fare	+0.255
Insight

Passengers paying higher fares generally survived more often.

This suggests:

Higher Fare
↓
Higher Social Status
↓
Higher Survival Rate
📈 Technologies Used
Python
Pandas
NumPy
Jupyter Notebook
