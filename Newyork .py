#!/usr/bin/env python
# coding: utf-8

# In[39]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


# In[40]:


df = pd.read_csv('NY-House-Dataset.csv')
df


# In[41]:


df.nunique()


# In[44]:


df = df.drop(columns = ['STATE','STREET_NAME'])
df.head()
     


# In[43]:



df = df.drop(columns = ['LATITUDE','LONGITUDE','BROKERTITLE','FORMATTED_ADDRESS','MAIN_ADDRESS','ADDRESS','LONG_NAME'])
df.head()


# In[27]:


X = df.drop(columns=['PRICE'])
Y = df['PRICE']
X
Y


# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5)


# In[29]:


from sklearn.linear_model import LinearRegression,LogisticRegression
model = LinearRegression()
model.fit(X_train,Y_train)


# In[30]:


a=model.score(X_train,Y_train)
print(a)


# In[31]:


model.score(X_test,Y_test)


# In[32]:


from sklearn.linear_model import Lasso
model1= Lasso(alpha = 50,max_iter = 100,tol = 0.1)


# In[33]:


model1.fit(X_train,Y_train)


# In[34]:


model1.score(X_train,Y_train)


# In[35]:


b=model1.score(X_test,Y_test)
print(b)


# In[36]:


from sklearn.linear_model import Ridge
model2  = Ridge(alpha = 1,max_iter = 1,tol = 0.001)


# In[37]:


model2.fit(X_train,Y_train)


# In[38]:


c=model2.score(X_train,Y_train)
c


# In[39]:


model2.score(X_test,Y_test)


# In[40]:


model3 = LogisticRegression()
model3.fit(X_train,Y_train)


# In[41]:


d=model3.score(X_train,Y_train)
d


# In[42]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


model4 = DecisionTreeClassifier()
model4.fit(X_train,Y_train)


# In[44]:


d=model4.score(X_train,Y_train)
d


# In[ ]:





# In[ ]:




