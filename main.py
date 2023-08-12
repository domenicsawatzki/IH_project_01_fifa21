#!/usr/bin/env python
# coding: utf-8

# Loading Data

# In[10]:


from my_modules import analysis_functions as mybib 
from my_modules import fifa_functions as fifa 

import pandas as pd 
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
import seaborn as sns 

# https://github.com/lilitrdavidyan 
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[8]:


raw_train_df = pd.read_csv('data/input/fifa21_train.csv')
# display(raw_train_df.describe().T)

raw_validate_df = pd.read_csv('data/input/fifa21_validate.csv')
# display(raw_validate_df.describe().T)


# In[3]:


# mybib.showUnique(raw_df)


# Remove Columns we don't need 

# In[11]:


# clean training dataset
clean_train_df = fifa.cleaning_process_for_fifa_dataset(raw_train_df)
display(clean_train_df)

# clean validate dataset
clean_validate_df = fifa.cleaning_process_for_fifa_dataset(raw_validate_df)
display(clean_validate_df)


# In[12]:


# split train dataset into X and y and fit the encoder 
y = clean_train_df['ova']
X = clean_train_df.drop(['ova'], axis=1)

# minMaxScaler
from sklearn.preprocessing import MinMaxScaler
MinMaxtransformer = MinMaxScaler().fit(X.select_dtypes(include = np.number))

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first')
encoder.fit(X.select_dtypes(include = object))

# split train dataset into X and y and fit the encoder 
y_validate = clean_validate_df['ova']
X_validate = clean_validate_df.drop(['ova'], axis=1)


# In[13]:


# normalize and encode X 
X = mybib.use_minMaxTransformer_and_oneHotEncoder(X, MinMaxtransformer, encoder)
X_validate = mybib.use_minMaxTransformer_and_oneHotEncoder(X_validate, MinMaxtransformer, encoder)


# In[14]:


# make a train test split
X_train, X_test, y_train, y_test = mybib.split_the_data_into_train_test_datasets(X, y, 0.2)


# In[15]:


# train model
from sklearn.linear_model import LinearRegression as LinReg

linreg=LinReg()    # model
linreg.fit(X_train, y_train)   # model train


# In[16]:


# validate datasets and print results for r2, MSE, RMSE, MAE
print('Train dataset')
mybib.predict_data_and_validate_model(X_train, y_train, linreg)

print('Test dataset')
mybib.predict_data_and_validate_model(X_test, y_test, linreg)

print('Validate dataset')
mybib.predict_data_and_validate_model(X_validate, y_validate, linreg)

