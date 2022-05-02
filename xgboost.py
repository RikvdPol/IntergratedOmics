#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[61]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[14]:


df = pd.read_csv(r"C:\Users\Farimah\Desktop\project\Covariates.csv",sep='\t' )
df.head(5)


# In[30]:


df.isna().sum()


# In[31]:


df = df.dropna(axis=0)


# In[32]:


df.isna().sum()


# In[37]:


df.dtypes


# In[ ]:


df.dty


# In[48]:


X = df[['Age', 'Gender']].values
y = df['BMI'].values


# In[50]:


X


# In[51]:


y


# In[84]:


# split data into train and test
seed = 1
test_size = 0.20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[85]:


from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
model = xgb.XGBRegressor()

scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error')
(-scores)**0.5


# In[86]:


# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores


# In[87]:


scores = abs(scores)
print('Mean MAE: {:.3f}'.format(scores.mean()))


# In[88]:


# another way
model = xgb.XGBRegressor()
model.fit(X_train, y_train)


# In[89]:


yhat = model.predict(X_test)


# In[90]:


score = abs(yhat - y_test)


# In[91]:


score.mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




