#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv("Cardiovascular_Disease_Dataset.csv")


# In[3]:


df


# In[4]:


df.columns


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x=df.drop(['target'],axis=1)
y=df['target']


# In[9]:


x


# In[10]:


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


# In[13]:


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




# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



model=LogisticRegression()
model.fit(x_train,y_train)
train_pred= model.predict(x_train)
test_pred= model.predict(x_test)

print("********* train data**********")


print("accuracy:",accuracy_score(y_train,train_pred))
print("precision:",precision_score(y_train,train_pred))
print("recall:",recall_score(y_train,train_pred))
print("f1score:",f1_score(y_train,train_pred))

print("********* test data**********")


print("accuracy:",accuracy_score(y_test,test_pred))
print("precision:",precision_score(y_test,test_pred))
print("recall:",recall_score(y_test,test_pred))
print("f1score:",f1_score(y_test,test_pred))

 


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



model=KNeighborsClassifier()
model.fit(x_train,y_train)
train_pred= model.predict(x_train)
test_pred= model.predict(x_test)

print("********* train data**********")


print("accuracy:",accuracy_score(y_train,train_pred))
print("precision:",precision_score(y_train,train_pred))
print("recall:",recall_score(y_train,train_pred))
print("f1score:",f1_score(y_train,train_pred))

print("********* test data**********")


print("accuracy:",accuracy_score(y_test,test_pred))
print("precision:",precision_score(y_test,test_pred))
print("recall:",recall_score(y_test,test_pred))
print("f1score:",f1_score(y_test,test_pred))


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



model=DecisionTreeClassifier()
model.fit(x_train,y_train)
train_pred= model.predict(x_train)
test_pred= model.predict(x_test)

print("********* train data**********")


print("accuracy:",accuracy_score(y_train,train_pred))
print("precision:",precision_score(y_train,train_pred))
print("recall:",recall_score(y_train,train_pred))
print("f1score:",f1_score(y_train,train_pred))

print("********* test data**********")


print("accuracy:",accuracy_score(y_test,test_pred))
print("precision:",precision_score(y_test,test_pred))
print("recall:",recall_score(y_test,test_pred))
print("f1score:",f1_score(y_test,test_pred))


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



model=RandomForestClassifier()
model.fit(x_train,y_train)
train_pred= model.predict(x_train)
test_pred= model.predict(x_test)

print("********* train data**********")


print("accuracy:",accuracy_score(y_train,train_pred))
print("precision:",precision_score(y_train,train_pred))
print("recall:",recall_score(y_train,train_pred))
print("f1score:",f1_score(y_train,train_pred))

print("********* test data**********")


print("accuracy:",accuracy_score(y_test,test_pred))
print("precision:",precision_score(y_test,test_pred))
print("recall:",recall_score(y_test,test_pred))
print("f1score:",f1_score(y_test,test_pred))


# In[ ]:




