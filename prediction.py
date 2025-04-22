import pandas as pd
from sklearn.ensemble import RandomForestClassifier
filePath = "./data_sets/titanic/test.csv"
test_df = pd.read_csv(filePath)
X_test = test_df.drop(columns=["Name", "Ticket", "Cabin"])  # Must match training features
model = RandomForestClassifier()
test_df["Survived"] = model.predict(X_test)
test_df[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)
