#!/usr/bin/env python
# coding: utf-8

# In[196]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[197]:


df = pd.read_csv("/Users/roua.chaouachi/Downloads/heart.csv")


# In[198]:


headers = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']


# In[199]:


df.head()
#one hot encoding in cp/ thall


# In[200]:


#Check the shape (size) of dataset
df.shape


# In[201]:


#Check the columns' data types
df.info()


# In[202]:


#Check the missing values
df.isnull().sum()


# In[203]:


#Check for the duplicates
df.duplicated().sum()


# In[204]:


df[df.duplicated()]


# In[205]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[206]:


print('Number of rows are',df.shape[0], 'and number of columns are ',df.shape[1])


# In[207]:


# get the information about dataframe
df.info()


# In[208]:


#Check number of unique values in each column
df.nunique()


# # Data Visualization 

# In[209]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True, cmap="RdBu")
plt.title("Correlation Between Variables", size=15)
plt.show()


# In[210]:


plt.figure(figsize=(10,6))
sns.countplot(df["sex"])
plt.title("Sex", size=15)
plt.show()


# In[211]:


plt.figure(figsize=(10,6))
sns.distplot(df["age"])
plt.title("Age", size=15)
plt.show()


# In[212]:


plt.figure(figsize=(10,6))
sns.distplot(df["chol"])
plt.title("Cholesterol", size=15)
plt.show()


# In[213]:


plt.figure(figsize=(10,6))
sns.countplot(df["cp"])
plt.title("Chest Pain Type", size=15)
plt.show()


# In[214]:


x=(df.restecg.value_counts())
print(x)
p = sns.countplot(data=df, x="restecg")
plt.show()


# In[215]:


x=(df.exng.value_counts())
print(x)
p = sns.countplot(data=df, x="exng")
plt.show()


# In[216]:


x=(df.thall.value_counts())
print(x)
p = sns.countplot(data=df, x="thall")
plt.show()


# In[217]:


plt.figure(figsize=(10,8))
sns.distplot(df[df["output"]==1]["age"], color="blue")
sns.distplot(df[df["output"]==0]["age"], color="red")
plt.title("Attack vs Age", size=15)
plt.show()


# In[218]:


plt.figure(figsize=(10,8))
sns.distplot(df[df["output"]==1]["chol"], color="blue")
sns.distplot(df[df["output"]==0]["chol"], color="red")
plt.title("Attack vs Cholesterol", size=15)
plt.show()


# In[219]:


plt.figure(figsize=(10,8))
sns.distplot(df[df["output"]==1]["trtbps"], color="blue")
sns.distplot(df[df["output"]==0]["trtbps"], color="red")
plt.title("Attack vs Resting Blood Pressure", size=15)
plt.show()


# In[220]:


plt.figure(figsize=(10,8))
sns.barplot(x=df["sex"], y=df["output"])
plt.title("Attack vs Sex", size=15)
plt.show()


# In[221]:


print(df[["sex", "output"]].groupby(['sex']).count())


# In[222]:


print(df[["cp", "output"]].groupby(['cp']).count())


# In[223]:


print(df[["fbs", "output"]].groupby(['fbs']).count())


# In[224]:


print (df[["exng", "output"]].groupby(['exng']).count())


# In[225]:


print (df[["exng", "output"]].groupby(['exng']).count())


# In[226]:


df1=df[df["output"] == 1]
sns.histplot(df1["thalachh"],bins=25, color="lightgreen");
plt.xlabel("Heart rate when outcome is 1")
plt.show()


# In[227]:


df2=df[df["output"]==0]
sns.histplot(df2["thalachh"],bins=25,  color="red");
plt.xlabel("Heart rate when outcome is 0")
plt.show()


# In[228]:


plt.figure(figsize=(13,13))
plt.subplot(2,3,1)

sns.violinplot(x = 'sex', y = 'output', data = df)
plt.subplot(2,3,2)

sns.violinplot(x = 'thall', y = 'output', data = df)
plt.subplot(2,3,3)

sns.violinplot(x = 'exng', y = 'output', data = df)
plt.subplot(2,3,4)

sns.violinplot(x = 'restecg', y = 'output', data = df)
plt.subplot(2,3,5)

sns.violinplot(x = 'cp', y = 'output', data = df)
plt.xticks(fontsize=9, rotation=45)
plt.subplot(2,3,6)

sns.violinplot(x = 'fbs', y = 'output', data = df)

plt.show()


# In[229]:


#restecg.head()
y = pd.get_dummies(df.cp, prefix='cp')
print(y.head())


# In[193]:


cp = pd.DataFrame({'cp': ['typical angina', 'atypical angina', 'non-anginal pain','asymptomatic']})
#restecg = pd.DataFrame({'restecg': ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy']})


# In[194]:


cp.head()


# In[156]:


df.drop(['cp'],axis=1, inplace=True)
#df.drop(['restecg'],axis=1, inplace=True)


# In[230]:


df = pd.concat([df,y],axis=1)
#df = pd.concat([df,pd.get_dummies(restecg['restecg'], prefix='restecg')],axis=1)


# In[231]:


df.head(16)


# In[143]:


#dummy_variable_1 = pd.get_dummies(df["cp"])
#dummy_variable_1.head()


# # Data processing 

# In[232]:


X = df.drop("output", axis=1)
y = df["output"]


# In[233]:


ss = StandardScaler()
X = ss.fit_transform(X)


# In[234]:


X = pd.DataFrame(X, columns=df.drop("output", axis=1).columns)
y = pd.DataFrame(y, columns=["output"])


# In[237]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True, random_state=42)


# In[238]:


models = pd.DataFrame(columns=["Model","Accuracy Score"])


# # Logistic regression model

# In[239]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
score = accuracy_score(y_test, predictions)
print("LogisticRegression: ", score)

new_row = {"Model": "LogisticRegression", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)


# # SVC model

# In[139]:


svm = SVC(random_state=0)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
score = accuracy_score(y_test, predictions)
print("SVC: ", score)

new_row = {"Model": "SVC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)


# # KNeighborsClassifier

# In[141]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
score = accuracy_score(y_test, predictions)
print("KNeighborsClassifier: ", score)


# In[151]:


score_list=[]

for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    predictions = knn2.predict(X_test)
    score_list.append(accuracy_score(predictions, y_test))


# In[152]:


plt.figure(figsize =(10, 6))
plt.plot(range(1, 20), score_list, marker ='o', markerfacecolor ='red', markersize = 10)
  
plt.title('Score vs K Value', size=15)
plt.xlabel('K value')
plt.ylabel('Score')


# In[149]:


knn3 = KNeighborsClassifier(n_neighbors=17)
knn3.fit(X_train, y_train)
predictions = knn3.predict(X_test)
score = accuracy_score(y_test, predictions)
print("KNeighborsClassifier: ", score)

new_row = {"Model": "KNeighborsClassifier", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)


# # Comparison of models

# In[153]:


models.sort_values(by="Accuracy Score", ascending=False)

