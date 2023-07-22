#!/usr/bin/env python
# coding: utf-8

# # IMPORTING REQUIRED LIBRARIES

# In[1]:


import pandas as pd
import numpy as np  


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# # READING DATA

# In[3]:


data1 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
data2 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")


# In[4]:


data1.head()


# In[5]:


data2.head()


# # MERGING TWO DATASETS

# In[6]:


data = pd.merge(data1, data2)
data.head()


# # DATA CLEANING

# In[7]:


data.isnull().sum()


# In[8]:


data.drop('Code',axis=1,inplace=True)


# In[9]:


data.head()


# In[10]:


data.size


# In[11]:


data.shape


# In[12]:


data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)


# In[13]:


data.head()


# # EXPLORATORY ANALYSIS 

# In[14]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.plot()


# In[15]:


sns.jointplot(x='Schizophrenia',y='mental_fitness',data=data,kind='reg',color='m')


# In[16]:


sns.jointplot(x='Bipolar_disorder',y='mental_fitness',data=data,kind='reg',color='red')


# In[19]:


sns.pairplot(data,corner=True)


# In[20]:


mean = data['mental_fitness'].mean()
mean


# In[21]:


fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()


# In[22]:


fig=px.bar(data.head(10),x='Year',y='mental_fitness',color='Year',template='ggplot2')
fig.show()


# # YEARWISE VARIATIONS IN MENTAL FITNESS OF DIFFERENT COUNTRIES

# In[23]:


fig = px.line(data, x="Year", y="mental_fitness", color='Country',markers=True,color_discrete_sequence=['red','blue'],template='plotly_dark')
fig.show()


# In[24]:


data = data.copy()


# In[25]:


data.head()


# In[26]:


data.info()


# In[28]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i]=l.fit_transform(data[i])


# In[29]:


X = data.drop('mental_fitness',axis=1)
y = data['mental_fitness']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# # LINEAR REGRESSION
# 

# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)


# In[33]:


ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)


# In[34]:


print("performance of train data")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[35]:


ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)


# In[36]:


print("perforamance of test data")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# # RANDOM FOREST REGRESSOR PERFORMS ON :
# 
# TRAINING AND TESTING DATA

# In[37]:


from sklearn.ensemble import RandomForestRegressor


# In[38]:


rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)


# In[39]:


ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)


# In[40]:


print("performance of train data")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[41]:


ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)


# In[42]:


print("performance of test train")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




