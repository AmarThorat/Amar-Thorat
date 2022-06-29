#!/usr/bin/env python
# coding: utf-8

# ## Use Random Forest to prepare a model on fraud data 
# ## treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[6]:


df = pd.read_csv("Fraud_check.csv")
df


# In[5]:


df.head()


# In[7]:


df.tail()


# In[10]:


df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


# In[11]:


df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
print(df)


# In[12]:


df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# In[13]:


df.tail()


# In[14]:


import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')


# In[15]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[16]:


df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[17]:


X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# In[20]:


df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"


# In[21]:


df.drop(["Taxable.Income"],axis=1,inplace=True)


# In[26]:


df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)


# In[28]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


# In[29]:


features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[30]:


colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[35]:


import warnings

warnings.filterwarnings("ignore")


# In[36]:


from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[37]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[38]:


model.n_outputs_


# In[39]:


model.oob_score_


# In[40]:


prediction = model.predict(x_train)


# In[41]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)


# In[42]:


np.mean(prediction == y_train)


# In[43]:


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[44]:


pred_test = model.predict(x_test)


# In[45]:


acc_test =accuracy_score(y_test,pred_test)


# In[50]:


get_ipython().system('pip install pydotplus')


# In[51]:


from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO


# In[52]:


tree = model.estimators_[5]


# In[56]:


dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)


# In[57]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[58]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[59]:


from sklearn import tree


# In[60]:


tree.plot_tree(model);


# In[61]:


colnames = list(df.columns)
colnames


# In[62]:


fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[63]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[64]:


preds


# In[65]:


pd.crosstab(y_test,preds)


# In[66]:


np.mean(preds==y_test)


# In[67]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[68]:


model_gini.fit(x_train, y_train)


# In[69]:


pred=model.predict(x_test)
np.mean(preds==y_test)


# In[70]:


from sklearn.tree import DecisionTreeRegressor
array = df.values
X = array[:,0:3]
y = array[:,3]


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[72]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[73]:


model.score(X_test,y_test)


# In[ ]:




