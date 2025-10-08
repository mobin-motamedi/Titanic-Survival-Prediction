# Baseline code with Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Random Forest instead of 
from sklearn.metrics import accuracy_score
from pathlib import Path


# Getting train.csv path
script_dir = Path(__file__).parent
data_path = script_dir.parent / 'data' / 'train.csv'

# 1. Loading the Data
data = pd.read_csv(data_path)

# 2. Handle missing values first of all
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 3. Feature Engineering

# Mapping Sex and Embarked to Numbers
data['Sex_num'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked_num'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Family number
data['FamilySize'] = data['SibSp'] + data['Parch']
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Cabin related
data['HasCabin'] = data['Cabin'].notnull().astype(int)
data['CabinLetter'] = data['Cabin'].str[0]
data['CabinLetter'] = data['CabinLetter'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}).fillna(0)

#Age groups
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 10, 18, 30, 40, 60, 80], labels=False)

# Interaction features
data['Sex_Class'] = data['Sex_num'] * data['Pclass']
data['Age_Class'] = data['Age'] * data['Pclass']

# 4. Select features
X = data[['Pclass', 'Sex_num', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_num','FamilySize', 'IsAlone', 'HasCabin', 'CabinLetter', 'AgeGroup', 'Sex_Class', 'Age_Class']]
y = data['Survived']

# 5. Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # random_state=42 rule

# 6. Train model using Random Forest

model = RandomForestClassifier( # arguments 2-7 are unnecessary and default, added to show how the model works on the baseline level
    n_estimators=200,      # Number of trees
    n_jobs=-1,              # Use all CPU cores
    random_state=42        # ofc
)

model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# Calculate feature correlations with target
feature_correlations = data[['Pclass', 'Sex_num', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_num','FamilySize', 'IsAlone', 'HasCabin', 'CabinLetter', 'AgeGroup', 'Sex_Class', 'Age_Class', 'Survived']].corr()['Survived'].drop('Survived').sort_values(ascending=False)

# 8. Output Evaluations
print("Model ran successfully\n")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Predictions: {(y_pred == y_val).sum()}/{len(y_val)} correct\n")
print("Feature correlations with survival:")
for feature, corr_value in feature_correlations.items():
    print(f"  {feature}: {corr_value:+.4f}")