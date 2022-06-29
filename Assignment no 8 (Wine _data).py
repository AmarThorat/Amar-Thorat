#!/usr/bin/env python
# coding: utf-8

# ## Perform Principal component analysis and perform clustering using first 
#  ## 3 principal component scores

# In[4]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[5]:


wine= pd.read_csv("wine.csv")


# In[6]:


print(wine.describe())
wine.head()


# In[7]:


wine['Type'].value_counts()


# In[8]:


Wine= wine.iloc[:,1:]
Wine


# In[9]:


Wine.shape


# In[10]:


Wine.info()


# In[11]:


wine_ary=Wine.values
wine_ary


# In[12]:


wine_norm=scale(wine_ary)
wine_norm


# In[13]:


pca = PCA()
pca_values = pca.fit_transform(wine_norm)
pca_values


# In[14]:


pca.components_


# In[15]:


var = pca.explained_variance_ratio_
var


# In[16]:


Var = np.cumsum(np.round(var,decimals= 4)*100)
Var


# In[17]:


plt.plot(Var,color="red");


# In[18]:


final_df=pd.concat([wine['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df


# In[19]:


import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df);


# In[20]:


sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type');


# In[21]:


pca_values[: ,0:1]


# In[22]:


x= pca_values[:,0:1]
y= pca_values[:,1:2]
plt.scatter(x,y);


# In[23]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[26]:


plt.figure(figsize=(12,10))
dendrogram=sch.dendrogram(sch.linkage(wine_norm,'complete'))


# In[25]:


hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters


# In[27]:


y=pd.DataFrame(hclusters.fit_predict(wine_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[28]:


wine2=wine.copy()
wine2['clustersid']=hclusters.labels_
wine2


# In[32]:


from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


# In[33]:


wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine_norm)
    wcss.append(kmeans.inertia_)


# In[31]:


plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');


# In[34]:


plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');


# In[35]:


clusters3=KMeans(3,random_state=30).fit(wine_norm)
clusters3


# In[36]:


clusters3.labels_


# In[37]:


wine3=wine.copy()
wine3['clusters3id']=clusters3.labels_
wine3


# In[38]:


wine3['clusters3id'].value_counts()


# In[ ]:




