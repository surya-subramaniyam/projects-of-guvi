#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt


# In[2]:


ad_data = pd.read_csv('D:\Python\csv file\house builder.csv')


# In[3]:


ad_data[:2]


# In[4]:


print(ad_data.head())
ad_data.info()


# In[5]:


ad_data.describe()


# In[6]:


p = sns.pairplot(ad_data)


# In[7]:


# visualize the relationship between the features and the response using scatterplots
p = sns.pairplot(ad_data, x_vars=['sqft_above','sqft_living15','sqft_lot15'], y_vars='price', size=7, aspect=0.7)


# In[8]:


ad_data.columns


# In[9]:


# Fitting the linear model

x = ad_data[['sqft_above','sqft_living15','sqft_lot15']] 
y = ad_data.price
x


# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.25)


# In[12]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X,y)
y_pred = regr.predict(X)
print(X[1:10])


# In[13]:


print("R squared: {}".format(r2_score(y_true=y,y_pred=y_pred)))


# In[15]:


residuals = y.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[16]:


p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-5,5)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')


# In[17]:


p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')


# In[18]:


plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(ad_data.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap


# In[ ]:




