#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression: Whether the client has subscribed a term deposit or not ?

# In[2]:


import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression


# In[5]:


app_log = pd.read_csv("bank-full.csv",sep=';')
app_log


# In[4]:


app_log.tail()


# In[6]:


app_log.columns


# In[7]:


columns = ['age', 'balance', 'duration', 'campaign', 'y']
app_log_sel = app_log[columns]
app_log_sel.info()


# In[8]:


pd.crosstab(app_log_sel.age,app_log_sel.y).plot(kind="line")


# ## the above graph shows that age group 20-60 has more rejection of application while 60-90 almost everybody

# In[10]:


sb.boxplot(data =app_log_sel,orient = "v")


# In[12]:


app_log_sel['outcome'] = app_log_sel.y.map({'no':0, 'yes':1})
app_log_sel.tail(10)


# In[13]:


app_log_sel.boxplot(column='age', by='outcome')


# In[14]:


feature_col=['age','balance','duration','campaign']
output_target=['outcome']
X = app_log_sel[feature_col]
Y = app_log_sel[output_target]


# In[15]:


classifier = LogisticRegression()


# In[16]:


classifier.fit(X,Y)


# In[17]:


classifier.coef_


# In[19]:


classifier.predict_proba (X)


# In[21]:


y_pred = classifier.predict(X)
y_pred


# In[25]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[26]:


confusion_matrix = confusion_matrix(Y,y_pred)
confusion_matrix 


# In[27]:


sb.heatmap(confusion_matrix, annot=True)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')


# In[ ]:




