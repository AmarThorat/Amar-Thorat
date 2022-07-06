#!/usr/bin/env python
# coding: utf-8

# ## 1) Prepare a classification Model using Naive Bayes 
# 
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


salary_train = pd.read_csv('SalaryData_Train(1).csv')
salary_test = pd.read_csv('SalaryData_Test.csv')


# In[23]:


salary_train.head()


# In[24]:


salary_test.head()


# In[4]:


salary_train.columns
salary_test.columns


# In[8]:


string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[9]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[10]:


for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])


# In[15]:


col_names=list(salary_train.columns)
train_X=salary_train[col_names[0:13]]
train_Y=salary_train[col_names[13]]
test_x=salary_test[col_names[0:13]]
test_y=salary_test[col_names[13]]


# ## Naive Bayes

# In[16]:


from sklearn.naive_bayes import GaussianNB
Gmodel=GaussianNB()


# In[18]:


train_pred_gau=Gmodel.fit(train_X,train_Y).predict(train_X)
test_pred_gau=Gmodel.fit(train_X,train_Y).predict(test_x)


# In[25]:


train_acc_gau=np.mean(train_pred_gau==train_Y)
test_acc_gau=np.mean(test_pred_gau==test_y)
train_acc_gau
test_acc_gau


# In[20]:


from sklearn.naive_bayes import MultinomialNB
Mmodel=MultinomialNB()
train_pred_multi=Mmodel.fit(train_X,train_Y).predict(train_X)
test_pred_multi=Mmodel.fit(train_X,train_Y).predict(test_x)


# In[21]:


train_acc_multi=np.mean(train_pred_multi==train_Y)
test_acc_multi=np.mean(test_pred_multi==test_y)
train_acc_multi
test_acc_multi

