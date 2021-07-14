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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow import keras


# In[2]:


df = pd.read_csv("DatasetTotal.csv", low_memory = False)


# In[3]:


df = df.drop([0,1])


# In[4]:


df['Timestamp']= pd.to_datetime(df['Timestamp'])


# In[5]:


df = df.sort_values(by=['Timestamp'])


# In[6]:


df = df.drop(columns = ['Timestamp'])


# In[7]:


input_label = np.array(df.loc[:, df.columns != "Label"]).astype(np.float)


# In[8]:


output_label = np.array(df["Label"])


# In[9]:


out = []
for o in output_label:
    if(o == "Benign"):out.append(0)
    else: out.append(1)
output_label = out


# In[10]:


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(input_label)
input_label = scaler.transform(input_label)


# In[11]:


inp = shuffle(input_label)


# In[12]:


inp_train,inp_test,out_train,out_test = train_test_split(input_label, input_label, test_size=0.2)


# In[13]:


input_model = keras.layers.Input(shape = (78,))
enc = keras.layers.Dense(units = 64, activation = "relu", use_bias = True)(input_model)
enc = keras.layers.Dense(units = 36, activation = "relu", use_bias = True)(enc)
enc = keras.layers.Dense(units = 18, activation = "relu")(enc)
dec = keras.layers.Dense(units = 36, activation = "relu", use_bias = True)(enc)
dec = keras.layers.Dense(units = 64, activation = "relu", use_bias = True)(dec)
dec = keras.layers.Dense(units = 78, activation = "relu", use_bias = True)(dec)
auto_encoder = keras.Model(input_model, dec)


# In[14]:


encoder = keras.Model(input_model, enc)
decoder_input = keras.layers.Input(shape = (18,))
decoder_layer = auto_encoder.layers[-3](decoder_input)
decoder_layer = auto_encoder.layers[-2](decoder_layer)
decoder_layer = auto_encoder.layers[-1](decoder_layer)
decoder = keras.Model(decoder_input, decoder_layer)


# In[15]:


auto_encoder.compile(optimizer=keras.optimizers.Adam(lr=0.00025), loss = "mean_squared_error", metrics = ['accuracy'])


# In[16]:


train = auto_encoder.fit(x = np.array(inp_train), y = np.array(out_train),validation_split= 0.1, epochs = 10, verbose = 2, shuffle = True)


# In[16]:


input_label = encoder.predict(input_label)


# In[17]:


inp = []
out = []
for i in range(len(input_label) - 12 + 1):
    aux = []
    for j in range(i, i + 12):
        aux.append(input_label[j])
    inp.append(aux)
    out.append(output_label[i + 12 - 1])


# In[18]:


input_label = None
output_label = None
gc.collect()


# In[ ]:


input_label = np.array(inp)
output_label = np.array(out)


# In[ ]:


inp = None
out = None
gc.collect()


# In[ ]:


def createModel():
    model = keras.Sequential([
        keras.layers.LSTM(units = 16, input_shape = (12,18), return_sequences = True, use_bias = True),
        keras.layers.LSTM(units = 8, return_sequences = False, use_bias = True),
        keras.layers.Dense(units = 2, activation = "softmax")
    ])
    model.compile(optimizer= keras.optimizers.Adam(lr= 0.00025), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


# In[ ]:


skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)


# In[ ]:


confusion_matrixs = []
roc_curvs = []


# In[ ]:


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


# In[ ]:


for i in range(10):
    print("------------------------------------")
    print("Modelo " + str(i))
    print(roc_curvs[i])
    print(confusion_matrixs[i])
    print("------------------------------------")

