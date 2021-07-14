#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import numpy as np
import gc
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow import keras


# In[2]:


df = pd.read_csv("../Dataset/01-03-2018.csv", low_memory = False)


# In[3]:


df = df.drop([0,1])


# In[4]:


input_label = np.array(df.loc[:, df.columns != "Label"]).astype(np.float)


# In[5]:


output_label = np.array(df["Label"])


# In[6]:


out = []
for o in output_label:
    if(o == "Benign"):out.append(0)
    else: out.append(1)
output_label = out


# In[7]:


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(input_label)
input_label = scaler.transform(input_label)


# In[8]:


input_label, output_label = shuffle(input_label, output_label)


# <h2>AutoEncoder</h2>

# In[9]:


inp_train,inp_test,out_train,out_test = train_test_split(input_label, input_label, test_size=0.2)


# In[10]:


input_model = keras.layers.Input(shape = (78,))
enc = keras.layers.Dense(units = 64, activation = "relu", use_bias = True)(input_model)
enc = keras.layers.Dense(units = 36, activation = "relu", use_bias = True)(enc)
enc = keras.layers.Dense(units = 18, activation = "relu")(enc)
dec = keras.layers.Dense(units = 36, activation = "relu", use_bias = True)(enc)
dec = keras.layers.Dense(units = 64, activation = "relu", use_bias = True)(dec)
dec = keras.layers.Dense(units = 78, activation = "relu", use_bias = True)(dec)
auto_encoder = keras.Model(input_model, dec)


# In[11]:


encoder = keras.Model(input_model, enc)
decoder_input = keras.layers.Input(shape = (18,))
decoder_layer = auto_encoder.layers[-3](decoder_input)
decoder_layer = auto_encoder.layers[-2](decoder_layer)
decoder_layer = auto_encoder.layers[-1](decoder_layer)
decoder = keras.Model(decoder_input, decoder_layer)


# In[12]:


auto_encoder.compile(optimizer=keras.optimizers.Adam(lr=0.00025), loss = "mean_squared_error", metrics = ['accuracy'])


# In[13]:


train = auto_encoder.fit(x = np.array(inp_train), y = np.array(out_train),validation_split= 0.1, epochs = 10, verbose = 2, shuffle = True)


# <h2>cross validation</h2>

# In[17]:


input_label = encoder.predict(input_label).reshape(len(input_label), 18, 1)


# In[18]:


def createModel():
    model = keras.Sequential([
        keras.layers.Conv1D(filters = 16, input_shape = (18,1), kernel_size = 3, padding = "same", activation = "relu", use_bias = True),
        keras.layers.MaxPool1D(pool_size = 3),
        keras.layers.Conv1D(filters = 8, kernel_size = 3, padding = "same", activation = "relu", use_bias = True),
        keras.layers.MaxPool1D(pool_size = 3),
        keras.layers.Flatten(),
        keras.layers.Dense(units = 2, activation = "softmax")
    ])
    model.compile(optimizer= keras.optimizers.Adam(lr= 0.00025), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


# In[19]:


skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)


# In[20]:


confusion_matrixs = []
roc_curvs = []


# In[21]:


for i, (train, test) in enumerate(skf.split(input_label, output_label)):
    print("Modelo " + str(i))
    inp_train, out_train = np.array(input_label)[train], np.array(output_label)[train]
    inp_test, out_test = np.array(input_label)[test], np.array(output_label)[test]
    model = createModel()
    model.fit(x = inp_train, y = out_train, validation_split= 0.1, epochs = 10, shuffle = True,verbose = 2)
    res = np.array([np.argmax(resu) for resu in model.predict(inp_test)])
    confusion_matrixs.append(confusion_matrix(out_test, res))
    fpr, tpr, _ = roc_curve(out_test,  res)
    auc = roc_auc_score(out_test, res)
    roc_curvs.append([fpr, tpr, auc])
    print("\n\n")


# <h2>Roc Curves</h2>

# In[ ]:


for i in range(10):
    print("------------------------------------")
    print("Modelo " + str(i))
    print(roc_curvs[i])
    print(confusion_matrixs[i])
    print("------------------------------------")

