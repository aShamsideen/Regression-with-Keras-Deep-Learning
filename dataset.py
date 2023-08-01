#!/usr/bin/env python
# coding: utf-8

# Importing the libraries to be used

# In[21]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Loading the dataset
# 
# The dataset is loaded and split into input and output variables

# In[22]:


df = pd.read_csv('housing.data.csv')
dataset = df.values
# split into x and y variable
x = dataset[:,0:13]
y = dataset[:,13]