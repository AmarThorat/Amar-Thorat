#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[6]:


sal_train=pd.read_csv('SalaryData_Test.csv')
sal_test=pd.read_csv('SalaryData_Train.csv')
sal_train.columns
sal_test.columns
string_col=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[7]:


sal_train


# In[8]:


sal_test


# In[10]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_col:
    sal_train[i]=label_encoder.fit_transform(sal_train[i])
    sal_test[i]=label_encoder.fit_transform(sal_test[i])


# In[11]:


train_x=sal_train.iloc[0:500,0:13]
train_y=sal_train.iloc[0:500,13]
test_x=sal_test.iloc[0:300,0:13]
test_y=sal_test.iloc[0:300,13]


# In[12]:


from sklearn.svm import SVC


# In[16]:


model_linear=SVC(kernel='linear')
model_linear.fit(train_x,train_y)
train_pred_lin=model_linear.predict(train_x)
test_pred_lin=model_linear.predict(test_x)
train_lin_acc=np.mean(train_pred_lin==train_y)
test_lin_acc=np.mean(test_pred_lin==test_y)


# In[17]:


train_lin_acc


# In[18]:


test_lin_acc


# In[20]:


model_poly=SVC(kernel='poly')
model_poly.fit(train_x,train_y)
train_pred_poly=model_poly.predict(train_x)
test_pred_poly=model_poly.predict(test_x)
train_poly_acc=np.mean(train_pred_poly==train_y)
test_poly_acc=np.mean(test_pred_poly==test_y)
train_poly_acc
test_poly_acc


# In[21]:


model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_x,train_y)
train_pred_rbf=model_rbf.predict(train_x)
test_pred_rbf=model_rbf.predict(test_x)
train_rbf_acc=np.mean(train_pred_rbf==train_y)
test_rbf_acc=np.mean(test_pred_rbf==test_y)
train_rbf_acc
test_rbf_acc


# In[ ]:




