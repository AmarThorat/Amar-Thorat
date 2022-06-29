#!/usr/bin/env python
# coding: utf-8

# ## ASSIGNMENT NO 1

# In[21]:


import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


# In[22]:


Q1_data = pd.read_csv("Cutlets.csv")
Q1_data.head()


# In[23]:


Q1_data.describe(include='all')


# In[24]:


Unit_A=Q1_data['Unit A'].mean()
Unit_B=Q1_data['Unit B'].mean()

print('Unit A Mean = ',Unit_A, '\nUnit B Mean = ',Unit_B)
print('Unit A Mean > Unit B Mean = ',Unit_A>Unit_B)


# In[25]:


sns.distplot(Q1_data['Unit A'])
sns.distplot(Q1_data['Unit B'])
plt.legend(['Unit A','Unit B'])


# In[26]:


sns.boxplot(data=[Q1_data['Unit A'],Q1_data['Unit B']],notch=True)
plt.legend(['Unit A','Unit B'])


# In[27]:


alpha=0.05
UnitA=pd.DataFrame(Q1_data['Unit A'])
UnitB=pd.DataFrame(Q1_data['Unit B'])
print(UnitA,UnitB)


# In[28]:


tStat,pValue =sp.stats.ttest_ind(UnitA,UnitB)
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


# In[29]:


if pValue <0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# 
# ## Inference is that there is no significant difference in the diameters of Unit A and Unit B

# ## Assignment no 2

# In[32]:


LabTAT =pd.read_csv('LabTAT.csv')
LabTAT.head()


# In[33]:


LabTAT.describe()


# In[34]:


Laboratory_1=LabTAT['Laboratory 1'].mean()
Laboratory_2=LabTAT['Laboratory 2'].mean()
Laboratory_3=LabTAT['Laboratory 3'].mean()
Laboratory_4=LabTAT['Laboratory 4'].mean()

print('Laboratory 1 Mean = ',Laboratory_1)
print('Laboratory 2 Mean = ',Laboratory_2)
print('Laboratory 3 Mean = ',Laboratory_3)
print('Laboratory 4 Mean = ',Laboratory_4)


# In[35]:


print('Laboratory_1 > Laboratory_2 = ',Laboratory_1 > Laboratory_2)
print('Laboratory_2 > Laboratory_3 = ',Laboratory_2 > Laboratory_3)
print('Laboratory_3 > Laboratory_4 = ',Laboratory_3 > Laboratory_4)
print('Laboratory_4 > Laboratory_1 = ',Laboratory_4 > Laboratory_1)


# In[36]:


sns.distplot(LabTAT['Laboratory 1'])
sns.distplot(LabTAT['Laboratory 2'])
sns.distplot(LabTAT['Laboratory 3'])
sns.distplot(LabTAT['Laboratory 4'])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[37]:


sns.boxplot(data=[LabTAT['Laboratory 1'],LabTAT['Laboratory 2'],LabTAT['Laboratory 3'],LabTAT['Laboratory 4']],notch=True)
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[38]:


alpha=0.05
Lab1=pd.DataFrame(LabTAT['Laboratory 1'])
Lab2=pd.DataFrame(LabTAT['Laboratory 2'])
Lab3=pd.DataFrame(LabTAT['Laboratory 3'])
Lab4=pd.DataFrame(LabTAT['Laboratory 4'])
print(Lab1,Lab1,Lab3,Lab4)


# In[39]:


tStat, pvalue = sp.stats.f_oneway(Lab1,Lab2,Lab3,Lab4)
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


# In[40]:


if pValue < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# Inference is that there no significant difference in the average TAT for all the labs.

# ## Assignment no 2

# In[42]:


BuyerRatio =pd.read_csv('BuyerRatio.csv')
BuyerRatio.head()


# In[43]:


BuyerRatio.describe()


# In[44]:


East=BuyerRatio['East'].mean()
West=BuyerRatio['West'].mean()
North=BuyerRatio['North'].mean()
South=BuyerRatio['South'].mean()

print('East Mean = ',East)
print('West Mean = ',West)
print('North Mean = ',North)
print('South Mean = ',South)


# In[45]:


sns.distplot(BuyerRatio['East'])
sns.distplot(BuyerRatio['West'])
sns.distplot(BuyerRatio['North'])
sns.distplot(BuyerRatio['South'])
plt.legend(['East','West','North','South'])


# In[46]:


sns.boxplot(data=[BuyerRatio['East'],BuyerRatio['West'],BuyerRatio['North'],BuyerRatio['South']],notch=True)
plt.legend(['East','West','North','South'])


# In[49]:


alpha=0.05
Male = [50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]
print(Sales)


# In[50]:


chiStats = sp.stats.chi2_contingency(Sales)
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')
if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[51]:


#critical value = 0.1
alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])# Find the critical value for 95% confidence*
                      #degree of freedom

observed_chi_val = chiStats[0]
#if observed chi-square < critical chi-square, then variables are not related
#if observed chi-square > critical chi-square, then variables are not independent (and hence may be related).
print('Interpret by critical value')
if observed_chi_val <= critical_value:
    # observed value is not in critical area therefore we accept null hypothesis
    print ('Null hypothesis cannot be rejected (variables are not related)')
else:
    # observed value is in critical area therefore we reject null hypothesis
    print ('Null hypothesis cannot be excepted (variables are not independent)')


# Inference : proportion of male and female across regions is same

# ## Assignment no 4

# In[52]:


Customer = pd.read_csv('Costomer+OrderForm.csv')
Customer.head()


# In[53]:


Customer.describe()


# In[54]:


Phillippines_value=Customer['Phillippines'].value_counts()
Indonesia_value=Customer['Indonesia'].value_counts()
Malta_value=Customer['Malta'].value_counts()
India_value=Customer['India'].value_counts()
print(Phillippines_value)
print(Indonesia_value)
print(Malta_value)
print(India_value)


# In[55]:


chiStats = sp.stats.chi2_contingency([[271,267,269,280],[29,33,31,20]])
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')
if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[56]:


#critical value = 0.1
alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
observed_chi_val = chiStats[0]
print('Interpret by critical value')
if observed_chi_val <= critical_value:
       print ('Null hypothesis cannot be rejected (variables are not related)')
else:
       print ('Null hypothesis cannot be excepted (variables are not independent)')


# In[ ]:




