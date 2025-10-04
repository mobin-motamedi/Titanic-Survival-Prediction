import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset
data = pd.read_csv('data/train.csv')

# Mapping Sex to 0 and 1 (Women had higher survivablity so maybe setting it to 1 helps)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Extract title from Name
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                        'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                        'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Map to numbers
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
data['Title'] = data['Title'].map(title_map)

# Create family size and IsAlone (after loading data)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# 2. Select features and target
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'IsAlone']]  # 6 features now (was 5)
y = data['Survived']  # Survived or not

# Fill missing Age values (leave the rest for now)
X = X.copy() # Don't ask why
X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())

# 3. Train-Test Split (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Make predictions and evaluate accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# 6. Print accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct predictions: {(y_pred == y_val).sum()} out of {len(y_val)}")
print(f"Wrong predictions: {(y_pred != y_val).sum()} out of {len(y_val)}")
print()