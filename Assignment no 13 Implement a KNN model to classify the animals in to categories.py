#!/usr/bin/env python
# coding: utf-8

# ## Implement a KNN model to classify the animals in to categories

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


Zoo=pd.read_csv('Zoo.csv')
zoo


# In[11]:


zoo=Zoo.iloc[:,1:]


# In[12]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(zoo,test_size=0.3,random_state=0)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier as KNC


# In[15]:


acc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])


# In[16]:


plt.plot(np.arange(3,50,2),[i[0] for i in acc],'bo-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'ro-')
plt.legend(['train','test'])


# In[17]:


neigh=KNC(n_neighbors=5)
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
train_acc=np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
test_acc=np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
train_acc
test_acc


# In[ ]:




