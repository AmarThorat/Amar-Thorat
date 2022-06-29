#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore")


# In[11]:


dataset=pd.read_csv('Salary_Data.csv')
dataset


# In[12]:


dataset.info()


# In[13]:


sns.distplot(dataset['YearsExperience'])


# In[14]:


sns.distplot(dataset['Salary'])


# In[15]:


dataset.corr()


# In[16]:


sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])


# In[17]:


model=smf.ols("Salary~YearsExperience",data=dataset).fit()


# In[18]:


model.tvalues, model.pvalues


# In[19]:


model.rsquared , model.rsquared_adj


# In[20]:


Salary = (25792.200199) + (9449.962321)*(3)
Salary


# In[21]:


new_data=pd.Series([3,5])
new_data


# In[22]:


data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred


# In[23]:


model.predict(data_pred)


# In[ ]:




