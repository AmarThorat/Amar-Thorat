#!/usr/bin/env python
# coding: utf-8

# ## classify the Size_Categorie using SVM

# In[8]:


import pandas as pd 
import numpy as np 
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


# In[9]:


forestfires = pd.read_csv("forestfires.csv")
forestfires


# In[10]:


data = forestfires.describe()


# In[11]:


forestfires.drop(["month","day"],axis=1,inplace =True)
forestfires


# In[14]:


predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]
predictors


# In[15]:


target


# In[16]:


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)


# In[18]:


fires = norm_func(predictors)
fires


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)


# In[20]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[21]:


model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)


# In[22]:


pred_test_linear = model_linear.predict(x_test)


# In[23]:



np.mean(pred_test_linear==y_test)


# In[24]:


acc = accuracy_score(y_test, pred_test_linear) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, pred_test_linear)


# In[25]:


model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)


# In[26]:


pred_test_poly = model_poly.predict(x_test)


# In[27]:


np.mean(pred_test_poly==y_test)


# In[28]:


model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)


# In[29]:


pred_test_rbf = model_rbf.predict(x_test)


# In[30]:


np.mean(pred_test_rbf==y_test) 


# In[31]:


model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)


# In[32]:


pred_test_sig = model_rbf.predict(x_test)


# In[33]:


np.mean(pred_test_sig==y_test)


# In[ ]:




