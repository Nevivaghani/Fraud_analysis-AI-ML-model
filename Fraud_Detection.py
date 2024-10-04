#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


df = pd.read_csv("C:/Users/NEVI/Downloads/Fraud.csv")


# In[14]:


df.head()


# In[15]:


df.info()


# In[16]:


df.isnull()


# In[17]:


# To display sum of null values from each columns
df.isnull().sum()


# In[18]:


# To display total sum of null values from whole dataset
df.isnull().sum().sum()


# In[19]:


# To drop nameOrig and nameDest columns from the dataset as it will not require further
df.drop(labels=["nameOrig" , 'nameDest'], inplace=True, axis=1)


# In[20]:


df.head()


# In[21]:


# To remane column newbalanceOrig to newbalanceOrg
df.rename(columns={'newbalanceOrig' : 'newbalanceOrg'}, inplace=True)


# In[22]:


df.head(2)


# In[23]:


# To print minimum values from each columns

df[['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']].min()


# In[24]:


# To print maximum values from each columns

df[['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']].max()


# In[25]:


# To check the transection type for each mathod with percentage

var = df.groupby('type').amount.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.pie(var, labels=var.index, autopct='%1.1f%%', startangle=90, counterclock=False)
ax1.set_title("Total amount per transaction type")
ax1.axis('equal')  
plt.show()


# In[26]:


# To know that as we saw that most tansactions are happaning by type of TRANSFER and CASH_OUT methods 
# so in which type fraud is happening.

df.loc[df.isFraud == 1].type.unique()


# #### As we can see maximum fraud activities are happening in transation types TRANSFER and CASH_OUT thus we will focus on these types only.

# In[27]:


sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap='coolwarm', annot=True)


# In[28]:


fraud = df.loc[df.isFraud == 1]
nonfraud = df.loc[df.isFraud == 0]


# In[29]:


fraudcount = fraud.isFraud.count()
nonfraudcount = nonfraud.isFraud.count()


# In[30]:


sns.heatmap(fraud.select_dtypes(include=['number']).corr(), cmap='coolwarm', annot=True)
plt.show()


# ##### As we can see,There are two noteworthy flags that capture my attention: the isFraud and isFlaggedFraud columns. From my understanding, isFraud serves as an indicator of actual fraudulent transactions, while isFlaggedFraud signifies transactions that have been prevented by the system due to certain thresholds being triggered. Based on the heatmap analysis, we observe a correlation between other columns and isFlaggedFraud, suggesting that there may also be a relationship with isFraud.

# In[31]:


print('Total number of Fraud transactions are : {}'.format(df.isFraud.sum()))
print('Total number of Fraud transactions which are marked as Fraud : {}'.format(df.isFlaggedFraud.sum()))
print('Ratio of fraud transaction to non-fraud transaction is : 1 / {}'.format(int(nonfraudcount//fraudcount)))

So, we can say that in 773 transactions 1 fraud transaction is happening 
# In[32]:


print('Amount lost by fraud transactions : {}' .format(int(fraud.amount.sum())))


# In[33]:


piedata = fraud.groupby(['isFlaggedFraud']).sum()
piedata


# In[34]:


f, axes = plt.subplots(1,1, figsize=(6,6))
axes.set_title("% of fraud transaction detected")
piedata.plot(kind='pie',y='isFraud',ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Detected','Detected'])
plt.show()


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_title("Fraud transaction which are Flagged Correctly")
axes.scatter(nonfraud['amount'],nonfraud['isFlaggedFraud'],c='b')
axes.scatter(fraud['amount'],fraud['isFlaggedFraud'],c='y')
plt.legend(loc='upper right',labels=['Not Flagged','Flagged'])
plt.show()


# The plot above clearly highlights the necessity for a fast and reliable system to identify fraudulent transactions. The current system allows some fraudulent transactions to bypass detection without being labeled as such. Conducting further data exploration could be beneficial in examining the relationships between the various features.

# ##### Data Exploration 

# In[37]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()


# In[38]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceDest'])
plt.show()


# In[39]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceDest'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()


# In[40]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
plt.show()


# ##### Data Cleaning

# In[41]:


import pickle


# In[42]:


df = df.replace(to_replace={'PAYMENT':1,'TRANSFER':2,'CASH_OUT':3,
                                            'CASH_IN':4,'DEBIT':5,'No':0,'Yes':1})


# In[43]:


df.head()


# In[44]:


x = df.drop(['isFraud'],axis=1)
y = df[['isFraud']]


# In[45]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[46]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 125)


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

clf = LogisticRegression(max_iter = 1000)

if True :
    probabilities = clf.fit(train_x, train_y.values.ravel()).predict_proba(test_x)[:,1]
    
if True: 
    print(average_precision_score(test_y,probabilities))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score

clf = KNeighborsClassifier(n_neighbors=5)
probabilities = clf.fit(train_x, train_y.values.ravel()).predict_proba(test_x)[:,1]
print(average_precision_score(test_y, probabilities))

