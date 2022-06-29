#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import sklearn 


# In[8]:


dataset = pd.read_csv('50_Startups.csv')
dataset


# In[6]:


dataset.head()


# In[9]:


dataset.tail()


# In[10]:


dataset.describe()


# In[11]:


print('There are ',dataset.shape[0],'rows and ',dataset.shape[1],'columns in the dataset.')


# In[12]:


dataset.isnull().sum()


# In[13]:


dataset.info()


# In[14]:


c = dataset.corr()
c


# In[19]:


sns.heatmap(c,annot=True,cmap='Oranges')
plt.show()


# In[20]:


outliers = ['Profit']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=dataset[outliers], orient="v", palette="Set2" , width=0.7) # orient = "v" : vertical boxplot , 
                                                                            # orient = "h" : hotrizontal boxplot
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")

plt.show()


# In[21]:


sns.boxplot(x = 'State', y = 'Profit', data = dataset)
plt.show()


# In[26]:


sns.distplot(dataset['Profit'],bins=5,kde=True)
plt.show()


# In[27]:


sns.pairplot(dataset)
plt.show()


# In[28]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X1 = pd.DataFrame(X)
X1.head()


# In[31]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)
x_train


# In[32]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)


# In[33]:


y_pred = model.predict(x_test)
y_pred


# In[34]:


testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)

training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)


# In[35]:


df = pd.DataFrame(data={'Predicted value':y_pred.flatten(),'Actual Value':y_test.flatten()})
df


# In[36]:


from sklearn.metrics import r2_score

r2Score = r2_score(y_pred, y_test)
print("R2 score of model is :" ,r2Score*100)


# In[37]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_pred, y_test)
print("Mean Squarred Error is :" ,mse*100)


# In[38]:


rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squarred Error is : ",rmse*100)


# In[39]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae)


# In[ ]:




