#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Importing the dataset

# In[4]:


dataset= pd.read_csv('Salary_Data.csv')
dataset.head()


# # Exploratory Data Analysis

# In[5]:


dataset.shape


# In[6]:


dataset.columns


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# # Extracting independent and dependent variables

# In[9]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[10]:


print(X.shape)


# In[11]:


print(y.shape)


# # Splitting the dataset into the Training set and Test set

# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)


# In[13]:


print(X_train.shape)


# # Training the Simple Linear Regression model on theTraining set

# In[14]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# # Getting the final linear regression equation with the values of the coefficients

# In[15]:


print("B1=",regressor.coef_)
print("B0 =",regressor.intercept_)


# # Predicting the Test set results
# 

# In[16]:


y_pred = regressor.predict(X_test)


# In[17]:


print(y_test)
print(y_pred)


# # Visualising the Training set results

# In[18]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
# regressor.predict(x_train) because it is predicted salaries for x_train
plt.title("Simple Linear Regression on Training Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# # Visualising the Test set results
# 

# In[19]:


plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
# y_pred because it is predicted salaries for x_test
plt.title("Simple Linear Regression on Testing Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# # Finding R^2 score

# In[20]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:




