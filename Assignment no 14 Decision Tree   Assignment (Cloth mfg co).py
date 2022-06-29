#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from  sklearn.preprocessing import LabelEncoder


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


company=pd.read_csv("Company_Data.csv")
company2=company.iloc[:,0:7]
company2


# In[6]:


labelenocer=LabelEncoder()
(company2['ShelveLoc'])=labelenocer.fit_transform(company2['ShelveLoc'])


# In[7]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[8]:


company2


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


df_norm = norm_func(company2.iloc[:,0:6])
df_norm.tail(10)


# In[11]:


company5=company2.iloc[:,6]
company5


# In[12]:


company3=pd.concat([df_norm,company5],axis=1)
company3


# In[13]:


x5=company3.iloc[:,0:6]
y5=company3.iloc[:,6]


# In[14]:


x5


# In[15]:


y5


# In[16]:


import seaborn as sns
sns.pairplot(data=company3, hue = 'ShelveLoc')


# In[17]:


Xtrain, Xtest, ytrain, ytest = train_test_split(x5,y5, test_size=0.2, random_state=0)


# In[19]:


model5=DecisionTreeClassifier(criterion ='entropy',max_depth=3)


# In[20]:


model5.fit(Xtrain,ytrain)


# In[21]:


preds5=model5.predict(Xtest)


# In[22]:


np.mean(preds5==ytest)*100


# In[23]:


print(classification_report(preds5,ytest))


# In[24]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[25]:


Kfold =    KFold(n_splits=10)
model3 =   RandomForestClassifier(n_estimators=100,max_features=3)
results=   cross_val_score(model5,x5,y5,cv=Kfold)
print(results.mean()*100)


# In[ ]:




