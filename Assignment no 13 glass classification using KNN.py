#!/usr/bin/env python
# coding: utf-8

# ## Prepare a model for glass classification using KNN
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


# In[2]:


df = pd.read_csv('glass.csv')
df.head()


# In[3]:


df.Type.value_counts()


# In[4]:


cor = df.corr()
sns.heatmap(cor)


# In[8]:


cor = df.corr()
sns.heatmap(cor)


# In[ ]:


sns.pairplot(df,hue='Type')
plt.show


# In[10]:


scaler = StandardScaler()


# In[12]:


scaler.fit(df.drop('Type',axis=1))


# In[13]:


StandardScaler(copy=True, with_mean=True, with_std=True)


# In[14]:


scaled_features = scaler.transform(df.drop('Type',axis=1))
scaled_features


# In[15]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[16]:


dff = df_feat.drop(['Ca','K'],axis=1) #Removing features - Ca and K 
X_train,X_test,y_train,y_test  = train_test_split(dff,df['Type'],test_size=0.3,random_state=45)


# In[17]:


knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')


# In[18]:


knn.fit(X_train,y_train)


# In[19]:


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')


# In[20]:


y_pred = knn.predict(X_test)


# In[21]:


print(classification_report(y_test,y_pred))


# In[22]:


accuracy_score(y_test,y_pred)


# In[ ]:




