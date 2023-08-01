#!/usr/bin/env python
# coding: utf-8

# In[11]:


from dataset import *


# ### Define a baseline model
# Crreate baseline model with a single fully connected hidden layer with the same number of neurons as input atttributes (13). The rectifier activation function is used for the hidden layer. No activation function is used for the output layer because it is a regression problem. ADAM optimization algorithm is used, and a mean squared error loss function is optimized. This will be the same metric to be used in evaluating the performance of the model. 
# The model is then evaluated using a 10-fold cross-validation.

# In[12]:


def baseline_model():
    
    # create model
    model = Sequential()
    model.add(Dense(13, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[13]:


estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x, y, cv=kfold, scoring='neg_mean_squared_error')
print('Baseline Model Evaluation: %.2f (%.2f) MSE' % (results.mean(), results.std()))


# ### Modelling the Standardized Dataset
# The input attributes of the dataset vary in their scale because they measure different quantities. Model will be re-evaluated using a standardized version of the dataset. Scikit learn Pipeline framework will be used to perform the standardization during the model evaluation process within each fold of the cross validation.

# In[14]:


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=kfold, scoring='neg_mean_squared_error', error_score='raise')
print('Standardized Evaluation: %.2f (%.2f) MSE' % (results.mean(), results.std()))


# ### Evaluate a Deeper Network Topology
# The Neural Network performance will be improved by adding more layers as this will allow the model to extract and recombine higher-order features embedded in the data.

# In[15]:


def deeper_model():
    model = Sequential()
    model.add(Dense(13, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[16]:


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=deeper_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=kfold, scoring='neg_mean_squared_error')
print('Deeper Model Evaluation: %.2f (%.2f) MSE' % (results.mean(), results.std()))


# ### Evaluate a Wider Network Topology
# the representational capacity of the model can be increased by creating a wider network. The effect of keeping a shallow network and increasing the number of neurons in one hidden layer will be evaluated.
# This will be done by increasing the number of neurons in the hidden layer from 13 to 20. 

# In[17]:


def wider_model():
    
    model = Sequential()
    model.add(Dense(20, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[18]:


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=kfold, scoring='neg_mean_squared_error', error_score='raise')
print('Wider Model Evaluation: %.2f (%.2f) MSE' % (results.mean(), results.std()))

