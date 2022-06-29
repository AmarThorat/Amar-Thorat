#!/usr/bin/env python
# coding: utf-8

# ## Perform Clustering for the crime data - Assignment no 7

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[3]:


data = pd.read_csv("crime_data.csv")
data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.isnull().any()


# In[7]:


mydata = data.drop(['Unnamed: 0'], axis=1)
mydata.head()


# In[8]:


scaler = MinMaxScaler()
norm_mydata = mydata.copy()
def minmaxscaler(x):
    for columnName, columnData in x.iteritems():
        x[columnName] = scaler.fit_transform(np.array(columnData).reshape(-1, 1))
    
minmaxscaler(norm_mydata)
norm_mydata.head()


# In[9]:


k = list(range(2,11))
sum_of_squared_distances = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(norm_mydata)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k, sum_of_squared_distances, 'go--')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of squares')
plt.title('Elbow Curve to find optimum K') 


# In[11]:


kmeans4 = KMeans(n_clusters = 4)

kmeans4.fit(norm_mydata)

y_pred = kmeans4.fit_predict(norm_mydata)
print(y_pred)

data['Cluster'] = y_pred+1 
data.head()


# In[12]:


centroids = kmeans4.cluster_centers_
centroids = pd.DataFrame(centroids, columns=['Murder', 'Assault', 'UrbanPop', 'Rape'])
centroids.index = np.arange(1, len(centroids)+1) # Start the index from 1
centroids


# In[13]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.set_palette("pastel")
sns.scatterplot(x=data['Murder'], y = data['Assault'], hue=data['Cluster'], palette='bright')


# In[14]:


data['Cluster'].value_counts()


# In[15]:


kmeans_mean_cluster = pd.DataFrame(round(data.groupby('Cluster').mean(),1))
kmeans_mean_cluster


# In[16]:


data[data['Cluster']==2]


# In[ ]:




