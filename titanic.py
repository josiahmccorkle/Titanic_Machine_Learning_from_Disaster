#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Example script with CLI flags")

# Add a flag (boolean)
parser.add_argument("--test", action="store_true", help="Enable this flag")

# Parse the arguments
args = parser.parse_args()

# List all CSV files in a folder
filePath = "./data_sets/titanic/"
files = [f for f in os.listdir(filePath) if f.endswith('.csv')]
print(files)
# # Load each file into a dictionary of DataFrames
dfs = {file: pd.read_csv(f"{filePath}{file}") for file in files}


if args.test:
    df = dfs["test.csv"]
else:
    df = dfs["train.csv"]

def preprocess(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    scaler = StandardScaler()
    for col in ['Age', 'Fare']:
        if col in df.columns:
            df[[col]] = scaler.fit_transform(df[[col]])
    return df

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

df = preprocess(df)

if not args.test:
    if df is not None:
        print(df.head())  # Display first few rows
    else:
        print("Error: 'train.csv' not found")

    # Compute correlation matrix
    corr_matrix = df.select_dtypes(include=['number']).corr()

    # Define features (X) and target (y)
    X = df.drop(columns=['Survived'])  # Features
    y = df['Survived']  # Target variable

    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    sns.countplot(data=df, x="Survived", hue="Sex")
    plt.show()
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm")
    plt.show()

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()
    sns.histplot(df['Age'].dropna(), bins=30, kde=True)
    plt.title("Age Distribution of Titanic Passengers")
    plt.show()

else:
    train_df = preprocess(dfs["train.csv"])
    # Align columns
    X_train = train_df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    y_train = train_df["Survived"]
    df = df[X_train.columns]

    # Train on full training set
    model = train_model(X_train, y_train)
    predictions = model.predict(df)

    # Save to CSV
    submission = pd.DataFrame({
        "PassengerId": dfs["test.csv"]["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")
