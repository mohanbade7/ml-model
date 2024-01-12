#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("Cardiovascular_Disease_Dataset.csv")


# In[2]:


df


# In[3]:


df.columns


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.columns


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x=df.drop(['target'],axis=1)
y=df['target']


# In[8]:


x


# In[9]:


y


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x=df.drop(['target'],axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("train mean squared error:",mean_squared_error(train_pred,y_train))
print("test mean squared error:",mean_squared_error(test_pred,y_test))


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

x=df.drop(['target'],axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("train mean squared error:",mean_squared_error(train_pred,y_train))
print("test mean squared error:",mean_squared_error(test_pred,y_test))
print("train mean squared error:",math.sqrt(mean_squared_error(train_pred,y_train)))
print("test mean squared error:",math.sqrt(mean_squared_error(test_pred,y_test)))


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error
import math

x=df.drop(['target'],axis=1)
y=df['target']

model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("************ linear Regression**************")

print("train mean squared errore:",mean_squared_error(train_pred,y_train))
print("test mean squared errore:",mean_squared_error(test_pred,y_test))
print("train mean squared errore:",math.sqrt(mean_squared_error(train_pred,y_train)))
print("test mean squared errore:",math.sqrt(mean_squared_error(test_pred,y_test)))

model=Lasso(alpha=2)
model.fit(x_train,y_train)
train_Pred=model.predict(x_train)
test_pred=model.predict(x_test)


print("************Lasso regularization************")
print("train mean squared errore:",mean_squared_error(train_pred,y_train))
print("test mean squared errore:",mean_squared_error(test_pred,y_test))
print("train mean squared errore:",math.sqrt(mean_squared_error(train_pred,y_train)))
print("test mean squared errore:",math.sqrt(mean_squared_error(test_pred,y_test)))

model=Ridge()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("***************** ridge*************")

print("train mean squared errore:",mean_squared_error(train_pred,y_train))
print("test mean squared errore:",mean_squared_error(test_pred,y_test))
print("train mean squared errore:",math.sqrt(mean_squared_error(train_pred,y_train)))
print("test mean squared errore:",math.sqrt(mean_squared_error(test_pred,y_test)))

model=ElasticNet()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("**********elastic*************")

print("train mean squared errore:",mean_squared_error(train_pred,y_train))
print("test mean squared errore:",mean_squared_error(test_pred,y_test))
print("train mean squared errore:",math.sqrt(mean_squared_error(train_pred,y_train)))
print("test mean squared errore:",math.sqrt(mean_squared_error(test_pred,y_test)))




# In[ ]:




