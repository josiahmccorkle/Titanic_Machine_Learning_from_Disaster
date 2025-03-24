#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

# List all CSV files in a folder
filePath = "./data_sets/titanic/"
files = [f for f in os.listdir(filePath) if f.endswith('.csv')]
print(files)
# # Load each file into a dictionary of DataFrames
dfs = {file: pd.read_csv(f"{filePath}{file}") for file in files}


# In[2]:


df = dfs.get("train.csv")  # Use train.csv instead
if df is not None:
    print(df.head())  # Display first few rows
else:
    print("Error: 'train.csv' not found")


# In[3]:


df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])  # Drop unnecessary columns
print(df.head())  # Check the new structure


# In[4]:


print(df.isnull().sum())  # Shows count of missing values per column


# In[5]:


df['Age'] = df['Age'].fillna(df['Age'].median())  # No warning


# In[6]:


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="Survived", hue="Sex")
plt.show()


# In[8]:


print(df.dtypes)
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.show()


# In[9]:


sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm")
plt.show()


# In[10]:


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})


# In[11]:


print(df["Sex"].unique())  # Should output: [0 1]


# In[12]:


df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


# In[14]:


import matplotlib.pyplot as plt  
import seaborn as sns  

sns.histplot(df['Age'].dropna(), bins=30, kde=True)  
plt.title("Age Distribution of Titanic Passengers")  
plt.show()  


# In[15]:


df["Embarked"]


# In[16]:


df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)


# In[17]:


print(df.head())  


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


# In[19]:


print(df[['Age', 'Fare']].head())


# In[20]:


df['Age'].std()


# In[ ]:





# In[ ]:





# In[ ]:




