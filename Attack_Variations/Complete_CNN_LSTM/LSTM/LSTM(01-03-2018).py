#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import itertools
import pandas as pd
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


dias = ["02-03-2018.csv", "21-02-2018.csv", "15-02-2018.csv", "16-02-2018.csv", "28-02-2018.csv", "14-02-2018.csv", "22-02-2018.csv", "23-02-2018.csv"]


# In[3]:


dfTotal = pd.DataFrame()
for dia in dias:
    dfAux = pd.read_csv("../../../Dataset/" + dia, low_memory = False)
    dfAux = dfAux.drop([0,1])
    dfTotal = pd.concat([dfTotal, dfAux])
    print("Ficheiro " + dia + " carregado")


# In[4]:


dfTotal


# In[5]:


dfAux = pd.read_csv("../../../Dataset/01-03-2018.csv", low_memory = False)


# In[6]:


dfAux = dfAux.drop([0,1])


# In[7]:


dfTeste = pd.DataFrame()
for colu in dfTotal.columns.tolist():
    dfTeste[colu] = dfAux[colu]


# In[8]:


dfAux = None


# In[9]:


dfTeste


# In[10]:


input_label_Total = []
output_label_Total= []
input_label_Teste = []
output_label_Teste = []


# In[11]:


input_label_Total = np.array(dfTotal.loc[:, dfTotal.columns != "Label"]).astype(np.float)
output_label_Total = np.array(dfTotal["Label"])
out = []
for o in output_label_Total:
    if(o == "Benign"):out.append(0)
    else: out.append(1)
output_label_Total = out


# In[12]:


dfTotal = None
gc.collect()


# In[13]:


input_label_Teste = np.array(dfTeste.loc[:, dfTeste.columns != "Label"]).astype(np.float)
output_label_Teste = np.array(dfTeste["Label"])
out = []
for o in output_label_Teste:
    if(o == "Benign"):out.append(0)
    else: out.append(1)
output_label_Teste = out


# In[14]:


dfTeste = None
out = None
gc.collect()


# In[15]:


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(np.concatenate((input_label_Total, input_label_Teste)))
input_label_Total = scaler.transform(input_label_Total)
input_label_Teste = scaler.transform(input_label_Teste)


# In[16]:


input_label_Total, output_label_Total = shuffle(input_label_Total, output_label_Total)
input_label_Teste, output_label_Teste = shuffle(input_label_Teste, output_label_Teste)


# In[17]:


encoder = keras.models.load_model('../Encoder.h5')


# In[18]:


input_label_Total = encoder.predict(np.array(input_label_Total))
input_label_Teste = encoder.predict(np.array(input_label_Teste))


# In[ ]:


tamanhoSequencia = 10


# In[ ]:


inp = []
out = []
num = 0
for i in range(len(input_label_Total) - tamanhoSequencia + 1):
    aux = []
    for j in range(i, i + tamanhoSequencia):
        aux.append(input_label_Total[j])
    inp.append(aux)
    out.append(output_label_Total[i + tamanhoSequencia - 1])
input_label_Total = inp
output_label_Total = out


# In[ ]:


inp = None
out = None


# In[19]:


model = keras.Sequential([
    keras.layers.LSTM(units = 16, input_shape = ((tamanhoSequencia,18)), return_sequences = True, use_bias = True),
    keras.layers.LSTM(units = 8, return_sequences = False, use_bias = True),
    keras.layers.Dense(units = 2, activation = "softmax")
])
model.compile(optimizer= keras.optimizers.Adam(lr= 0.00025), loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# In[20]:


model.fit(x = np.array(input_label_Total), y = np.array(output_label_Total), validation_split= 0.1, epochs = 10, shuffle = True,verbose = 1)


# In[ ]:


inp = []
out = []
num = 0
for i in range(len(input_label_Teste) - tamanhoSequencia + 1):
    aux = []
    for j in range(i, i + tamanhoSequencia):
        aux.append(input_label_Teste[j])
    inp.append(aux)
    out.append(output_label_Teste[i + tamanhoSequencia - 1])
input_label_Teste = inp
output_label_Teste = out


# In[ ]:


inp = None
out = None


# In[21]:


res = [np.argmax(resu) for resu in model.predict(input_label_Teste)]


# In[22]:


cm = confusion_matrix(y_true = np.array(output_label_Teste).reshape(len(output_label_Teste)), y_pred = np.array(res))


# In[ ]:


print(cm)
print("\n")


# In[ ]:


output_label_Teste = np.array(output_label_Teste).reshape(len(output_label_Teste))
res = np.array(res)
fpr, tpr, _ = roc_curve(output_label_Teste,  res)
auc = roc_auc_score(output_label_Teste, res)


# In[ ]:


print(str(fpr) + " | " + str(tpr) + " | " + str(auc))

