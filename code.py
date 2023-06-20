#!/usr/bin/env python
# coding: utf-8

# In[201]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jupyterthemes import jtplot
jtplot.style(theme="monokai", context="notebook", ticks=True, grid=False)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[175]:


# read data from the given dataset
data = pd.read_csv("kyphosis.csv")


# In[176]:


# visualizing read data
data


# In[177]:


# getting last 10 data from the data set
data.tail(10)


# In[178]:


# getting a summary of our data set
data.info()


# In[179]:


# getting some statistical parameters on the given data set
print("mean", data["Age"].mean()/12)
print("min", data["Age"].min()/12)
print("max", data["Age"].max()/12) # /12 yse to convert age value from months to years
print()
data.describe()


# In[180]:


# preprocessing data of the dataset(we have to convert non numerical data into numerical data)
LabelEncoder_Y = LabelEncoder()
data['Kyphosis'] = LabelEncoder_Y.fit_transform(data['Kyphosis'])
data


# In[181]:


# Defining two data frames based on the 'Kyphosis' column value
# Filtering the 'data' DataFrame to create a new DataFrame 'kyphosis_true' that contains rows where 'Kyphosis' is equal to 1
kyphosis_true = data[data['Kyphosis']==1]
# Filtering the 'data' DataFrame to create a new DataFrame 'kyphosis_false' that contains rows where 'Kyphosis' is equal to 0
kyphosis_false = data[data['Kyphosis']==0]


# In[182]:


# getting precentage values for each outcome in the data set
print("Disease precentage after operation: ", (len(kyphosis_true)/len(data))*100,"%")


# In[183]:


plt.figure(figsize=(10, 10))

# Generating a heatmap of the correlation matrix of the 'data' DataFrame
# The `corr()` function computes pairwise correlation of columns, by default using Pearson correlation coefficient
# The resulting correlation matrix is then visualized as a heatmap
# The `annot=True` parameter adds the correlation values as annotations to the heatmap cells
sns.heatmap(data.corr(), annot = True)


# In[184]:


# Generating a pair plot using the seaborn library for the 'data' DataFrame
# The 'hue' parameter is set to "Kyphosis" to add color differentiation based on the "Kyphosis" column
sns.pairplot(data, hue = "Kyphosis")


# In[185]:


# plotting the count of the each outcome of the dataset
sns.countplot(x = data['Kyphosis'])


# In[186]:


# deop the target variable and assigning it into y
X = data.drop(["Kyphosis"], axis=1)
Y = data["Kyphosis"]
print('x values')
print(X)
print()
print('Y values')
print(Y)


# In[187]:


# in this project I split my data, 80% to training and the 20% to the testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[188]:


# printing the shapes of splited data
X_train.shape


# In[189]:


X_test.shape


# In[190]:


# scaling the data 
# for logistic regression, the scalig is not needed
# but when we implement keras or deep neural network training
# sc = StandradScalaer()
#X_train = sc.fit_trans(X_train)
# X_test = sc.transform(X_test)


# In[191]:


# training the model using logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)


# In[193]:


# evaluatng proformance of the trained model

# evauating performance using confusion metrics
Y_predict_test = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict_test)
sns.heatmap(cm, annot=True)


# In[194]:


# getting the classification report
print(classification_report(Y_test, Y_predict_test))


# In[195]:


# getting decision tree classification for the above data set to improve the performance of the model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)


# In[196]:


Y_predict_tree = decision_tree.predict(X_test)
cm_tree = confusion_matrix(Y_test, Y_predict_tree)
sns.heatmap(cm_tree, annot=True)


# In[197]:


# getting classification reprot for the decision tree model
print(classification_report(Y_test, Y_predict_tree))


# In[200]:


# Implementing feature importance to find the most critical feature
feature_importance = pd.DataFrame(decision_tree.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importance)


# In[202]:


# implementing random forest classifier model to improve the performance of the model
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, Y_train)


# In[203]:


# visualizing the model using a heatmap
Y_predict_forest = RandomForest.predict(X_test)
cm_forest = confusion_matrix(Y_test, Y_predict_forest)
sns.heatmap(cm_forest, annot=True)


# In[204]:


# getting classification reprot for the decision random forest model
print(classification_report(Y_test, Y_predict_forest))


# In[ ]:




