#!/usr/bin/env python
# coding: utf-8

# # Import Dependencies

# In[75]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


# # Data Collection and Processing

# In[3]:


#loading the dataset to pandas DataFrame
df=pd.read_csv("loan-dataset.csv")


# In[4]:


type(df)


# In[5]:


# printing the first 5 rows of the dataframe
df.head()


# In[6]:


# number of rows and columns
df.shape


# In[7]:


# statistical measures
df.describe()


# In[8]:


# number of missing values in each column
df.isnull().sum()


# In[9]:


# filling the missing values
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()


# In[10]:


# label encoding
df.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[11]:


df.head()


# In[12]:


# Dependent column values
df['Dependents'].value_counts()


# In[13]:


# replacing the value of 3+ to 4
df=df.replace(to_replace='3+',value=4)


# In[14]:


df['Dependents'].value_counts()


# # Data Visualization

# In[16]:


# education and loan status
sns.countplot(x='Education',hue='Loan_Status',data=df)


# In[17]:


# marital status and loan status
sns.countplot(x='Married',hue='Loan_Status',data=df)


# In[18]:


# Dependents and loan status
sns.countplot(x='Dependents',hue='Loan_Status',data=df)


# In[19]:


# Gender and loan status
sns.countplot(x='Gender',hue='Loan_Status',data=df)


# In[24]:


# convert categorical columns to numerical values
df.replace({"Married":{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Urban':2,'Semiurban':1,'Rural':0},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[25]:


df.head()


# In[27]:


# seperating the data and label
X=df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=df['Loan_Status']


# In[28]:


print(X)
print(Y)


# # Train Test Split

# In[29]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[30]:


print(X.shape,X_train.shape,X_test.shape)


# # Training the model:

# # Support Vector Machine Model

# In[31]:


classifier=svm.SVC(kernel='linear')


# In[33]:


# training the support vector machine model
classifier.fit(X_train,Y_train)


# # Model Evaluation

# In[35]:


# accuracy score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[36]:


print("accuracy on training data:",training_data_accuracy)


# In[37]:


# accuracy score on test data
X_test_prediction=classifier.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[38]:


print("accuracy on test data:",testing_data_accuracy)


# # Random Forest Classification Model

# In[48]:


rf_clf=RandomForestClassifier()


# In[49]:


# training the Random Forest Classification Model
rf_clf.fit(X_train,Y_train)


# # Model Evaluation

# In[50]:


# accuracy score on training data
X_train_prediction=rf_clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[51]:


print("acc of random forest clf is",training_data_accuracy)


# In[52]:


# accuracy score on testing data
X_test_prediction=rf_clf.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[53]:


print("acc of random forest clf is",testing_data_accuracy)


# # Naive Bayes Model

# In[54]:


nb_clf=GaussianNB()


# In[55]:


# training the Naive Bayes Model
nb_clf.fit(X_train,Y_train)


# # Model Evaluation

# In[56]:


# accuracy score on training data
X_train_prediction=nb_clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[57]:


print("acc of naive bayes clf is",training_data_accuracy)


# In[58]:


# accuracy score on testing data
X_test_prediction=nb_clf.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[59]:


print("acc of naive bayes clf is",testing_data_accuracy)


# # Decision Tree Classification Model

# In[61]:


dt_clf=DecisionTreeClassifier()


# In[62]:


# training the Decision Tree Model
dt_clf.fit(X_train,Y_train)


# # Model Evaluation

# In[63]:


# accuracy score on training data
X_train_prediction=dt_clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[64]:


print("acc of decision tree clf is",training_data_accuracy)


# In[65]:


# accuracy score on testing data
X_test_prediction=dt_clf.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[66]:


print("acc of decision tree clf is",testing_data_accuracy)


# # KNeighbors Classification Model

# In[69]:


kn_clf=KNeighborsClassifier()


# In[70]:


# training the KNeighbors Model
kn_clf.fit(X_train,Y_train)


# # Model Evaluation

# In[71]:


# accuracy score on training data
X_train_prediction=kn_clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[72]:


print("acc of KNeighbors clf is",training_data_accuracy)


# In[73]:


# accuracy score on testing data
X_test_prediction=kn_clf.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[74]:


print("acc of KNeighbors clf is",testing_data_accuracy)


# # Gradient Boosting Classification Model

# In[76]:


gb_clf=GradientBoostingClassifier()


# In[77]:


# training the Gradient Boosting Model
gb_clf.fit(X_train,Y_train)


# # Model Evaluation

# In[78]:


# accuracy score on training data
X_train_prediction=gb_clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[79]:


print("acc of Gradient Boosting clf is",training_data_accuracy)


# In[80]:


# accuracy score on testing data
X_test_prediction=gb_clf.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[81]:


print("acc of Gradient Boosting clf is",testing_data_accuracy)


# In[ ]:




