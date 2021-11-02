#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import os 
import warnings
warnings.filterwarnings('ignore')


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[54]:


data_set = pd.read_csv(r"gld_price_data.csv")


# In[55]:


data_set.head()


# In[56]:


data_set.shape


# In[57]:


data_set.isnull().sum()


# In[58]:


data_set["Date"].value_counts()


# In[59]:


data_set["Date"].nunique()


# In[60]:


data_set.info()


# In[61]:


data_set.describe()


# In[62]:


data_set.head()


# In[73]:


data_set["Month"]= [i.split("/")[0]for i in data_set.Date]
data_set["Day"]= [i.split("/")[1]for i in data_set.Date]
data_set["Year"]= [i.split("/")[2]for i in data_set.Date]


# In[ ]:





# In[76]:


data_set.head()


# In[67]:


max(data_set["Day"])


# In[68]:


max(data_set["Month"])


# In[69]:


max(data_set["Year"])


# In[70]:


data_set.info()


# In[71]:


data_set.describe()


# In[77]:


convert_dict = {'Month': int,
                'Day': int,
                'Year': int
               }
  
data_set = data_set.astype(convert_dict)
print(data_set.dtypes)


# In[79]:


print(data_set.info())


# In[80]:


data_set.describe()


# In[82]:



correlation_dataset = data_set.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_dataset,cbar = True, square=True, fmt='.1g',annot= True,annot_kws = {'size':8},cmap ="Blues")


# In[84]:


data_set["Year"].value_counts()


# In[85]:


data_set["Year"].nunique()


# In[83]:


f,ax = plt.subplots(figsize=(10,10))
sns.set(style="ticks")
sns.barplot(x="Year",y="EUR/USD", data= data_set,label="Price vs Year",color='b')
sns.set(context="paper")
ax.legend(ncol=2,loc="upper right", frameon=True)


# In[95]:


f,ax = plt.subplots(figsize=(10,10))
sns.set(context="paper")
sns.set(style="ticks")
sns.scatterplot(x="USO",y="EUR/USD",data=data_set,color='red',label="Price vs USO")
ax.legend(loc="upper right",frameon="True")


# In[97]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[100]:


data_set = data_set.drop("Date",axis =1)


# In[101]:


x= data_set.drop("EUR/USD",axis =1)
y = data_set["EUR/USD"]


# In[102]:


x.head()


# In[103]:


y.head()


# In[108]:


X_train, X_test, Y_train, Y_test  =  train_test_split(x, y, test_size=0.10, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[115]:


from sklearn.metrics import mean_squared_error


# In[117]:


rmses = []
degress = np.arange(1, 10)
min_rmse, min_deg = 1e10, 0
for deg in degress:
    poly_features = PolynomialFeatures(degree =deg, include_bias=False )
    X_poly = poly_features.fit_transform(X_train)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, Y_train)

    x_poly_test = poly_features.fit_transform(X_test)
    prediction_test = poly_reg.predict(x_poly_test)

    poly_mse = mean_squared_error(Y_test, prediction_test)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)

    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg


print("Rmse:", min_rmse)
print("best Degree of polynomial : ",min_deg)


# In[119]:


poly = PolynomialFeatures(degree=5)
X_polynom_train = poly.fit_transform(X_train)

poly_reg= LinearRegression()
poly_reg.fit(X_polynom_train, Y_train)

X_polynom_test = poly.fit_transform(X_test)
Y_pred_poly = poly_reg.predict(X_polynom_test)



# In[120]:


poly_mse = mean_squared_error(Y_test, Y_pred_poly)
poly_rmse = np.sqrt(poly_mse)

print('Rmse = ',poly_rmse)


# In[121]:


from sklearn.metrics import r2_score


# In[122]:


R2_score =  r2_score(Y_test, Y_pred_poly)
print("R-Squared = ",R2_score)


# In[ ]:




