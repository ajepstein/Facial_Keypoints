#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:38:09 2020

@author: adamepstein
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataDir = '/Users/adamepstein/Documents/facial-keypoints-detection/'

df_train = pd.read_csv(dataDir + 'training.csv')
df_test = pd.read_csv(dataDir + 'test.csv')
df_lookup = pd.read_csv(dataDir + 'IdLookupTable.csv')

N_train = df_train.shape[0]
N_test = df_test.shape[0]
numTargets = df_train.shape[1] - 1   

numRows = 96
numCols = 96
X_train = np.zeros((N_train,numRows,numCols))
X_test =  np.zeros((N_test,numRows,numCols))


#%%
#Y_train = df_train.values[:,0:numTargets]

Y_train = df_train.drop('Image',axis = 1)
Y_train.fillna(Y_train.mean(),inplace = True)
Y_train = Y_train.values
#Y_train = np.float64(Y_train)

for i in range(N_train):
    
    pixelVals = np.uint8(df_train['Image'][i].split())
    X_train[i] = np.reshape(pixelVals,(numRows,numCols))
    
    if i % 100 == 0:
        print(i)
        
    Yi = Y_train[i]
    xCoords = [Yi[j] for j in range(numTargets) if j % 2 == 0]
    yCoords = [Yi[j] for j in range(numTargets) if j % 2 == 1]
    
#    plt.imshow(X_train[i], cmap = 'gray')
#    plt.scatter(x=xCoords, y=yCoords, c='r', s=40)
#    plt.pause(5)

#%%  
for i in range(N_test):
    
    pixelVals = np.uint8(df_test['Image'][i].split())
    X_test[i] = np.reshape(pixelVals,(numRows,numCols))
    
    if i % 100 == 0:
        print('Constructing X_test. i is: ',i)
    

    
X_train = X_train/255.0
X_test = X_test/255.0
X_train = np.reshape(X_train, (N_train, 96, 96, 1))
X_test = np.reshape(X_test, (N_test, 96, 96, 1))

#%%
X_train_reduced, X_val, Y_train_reduced, Y_val = train_test_split(X_train, Y_train, test_size = .2)


#20% of x_train should go into validation data set
#80% of x_train should go into new training data set
#%%
#Changed my dimensions in each layer
#also changed my batch_size to 150

#I do not remember original architecture
#Changing the architecture improved the runt time speed and score

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (96, 96, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(30)
    ])

model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
#%%
#went from 2 to 10 to 50 epochs
#Training 50 epochs is very slow
numEpochs = 50
history = model.fit(X_train_reduced, Y_train_reduced, epochs = numEpochs, batch_size = 150, 
                    validation_data = (X_val, Y_val))
#%%
history_dict = history.history
#%%
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_mae = history_dict['mae']
val_mae = history_dict['val_mae']

#%%
epochs = np.arange(1, numEpochs + 1)
#%% Plotting Training & Validation Loss over each epoch
plt.figure()
plt.plot(epochs, train_loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#%% Plotting Training & Validation MAE over each epoch
plt.figure()
plt.plot(epochs, train_mae, 'bo', label = 'Training Mean Absolute Error')
plt.plot(epochs, val_mae, 'b', label = 'Validation Mean Absolute Error')
plt.title('Training & Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
#%%
Y_pred = model.predict(X_test)
#%%
d = {}
numRows_lookup = df_lookup.shape[0]
cols = df_train.columns
cols = cols[:-1]

for i in range(len(cols)):
    column_name = cols[i]
    d[column_name] = i
#%%
preds = np.zeros(numRows_lookup)
for i in range(numRows_lookup):
    feature_name = df_lookup['FeatureName'][i]
    col = d[feature_name]
    image_index = df_lookup['ImageId'][i] - 1
    preds[i] = Y_pred[image_index, col]
    #This kaggle contest requires predictions to be between 0 and 96 inclusive
    if preds[i] > 96:
        preds[i] = 96
    if preds[i] < 0:
        preds[i] = 0
        
    #preds is a subset of the total predictions
    #preds contains the predictions that kaggle asks for

#%%
for i in range(50):
    Xi = X_test[i]
    Xi = np.reshape(Xi, (96,96))
    plt.imshow(Xi, cmap ='gray')
    
    Yi = Y_pred[i]
    Yi = np.reshape(Yi, (15, 2))
    plt.scatter(Yi[:,0], Yi[:,1])
    plt.pause(0.5)


#%% Kaggle Submission
#First Kaggle Score: 9.85713
#Second Kaggle Score: 5.86437
#Third Kaggle Score: 3.75705
  
kaggleSubmission = df_lookup[['RowId']]
kaggleSubmission.loc[:,'Location'] = preds

kaggleSubmission.to_csv('/Users/adamepstein/Documents/facial-keypoints-detection/kaggleFacialKeypointsSubmission.csv',index = False)    
    
    
#In my writeup include the 2 graphs plus the good predictions and not so good predictions
#3 of each

