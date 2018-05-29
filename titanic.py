
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')


# In[17]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[18]:


# Importing the dataset
dataset = pd.read_csv('train_titanic.csv')
predict = pd.read_csv('test_titanic.csv')
iv = dataset.iloc[:,[2,4,5,6,7,9,11]].values
iv_test = predict.iloc[:,[1,3,4,5,6,8,10]].values
dv = dataset.iloc[:, 1].values

iv_test[1]


# In[19]:


#Encoding
#Label encoding
#Import Label Encoder
from sklearn.preprocessing import LabelEncoder
encode_iv1 = LabelEncoder()#
iv[:,1] = encode_iv1.fit_transform(iv[:,1])
iv[:,-1] = encode_iv1.fit_transform(iv[:,-1].astype(str))
iv_test[:,1] = encode_iv1.fit_transform(iv_test[:,1])
iv_test[:,-1] = encode_iv1.fit_transform(iv_test[:,-1].astype(str))


# In[20]:


#Imputation
#Import Impute library
from sklearn.preprocessing import Imputer
#Clear an instance of the class Imputer
imputer_mean = Imputer(missing_values="NaN",strategy="median",axis=0)
imputer_mode = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
iv[:,[2]]=imputer_mean.fit_transform(iv[:,[2]])
iv[:,[-1]]=imputer_mode.fit_transform(iv[:,[-1]])
iv_test[:,[2]]=imputer_mean.fit_transform(iv_test[:,[2]])
iv_test[:,[-2]]=imputer_mean.fit_transform(iv_test[:,[2]])
iv_test[:,[-1]]=imputer_mode.fit_transform(iv_test[:,[-1]])


# In[21]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv_train = sc.fit_transform(iv)
iv_test = sc.transform(iv_test)


# In[24]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10,criterion='entropy', random_state = 0,n_jobs = 10000)
rf_classifier.fit(iv_train, dv)
survivors = rf_classifier.predict(iv_test)


# In[23]:


prediction = pd.read_csv("tit_predict.csv")
prediction['Survived'] = survivors
prediction.to_csv('titanic_predicted.csv', sep=',')
prediction

