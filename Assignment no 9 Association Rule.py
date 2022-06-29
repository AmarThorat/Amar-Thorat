#!/usr/bin/env python
# coding: utf-8

# ## Prepare rules for the all the data sets 

# In[3]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[5]:


book=pd.read_csv('book.csv')


# In[6]:


book.head()


# In[9]:


frequent_itemsets = apriori(book, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[ ]:


rules=association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules

