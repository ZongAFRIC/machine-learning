#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:02:45 2019

@author: nicolas
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD
print("                                                        ")
print("                                                        ")
print("                                                        ")
print("=======>       APPRENTISSAGE AVEC KERAS       <=======")
print("                                                        ")
nb_iteration = int(input("Entrer le nombre d'itération: "))
pas = float(input("Entrer le pas: "))
model = tf.keras.Sequential([
 tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,), use_bias=True)
])

sgd = SGD(pas)


model.compile(optimizer='sgd',
 loss='binary_crossentropy',
 metrics=['accuracy'])

data_x = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]) #donnée d'apprentissage: x
data_y = np.array([[1], [1], [1], [1]]) #donnée d'apprentissage:
model.fit(data_x, data_y, epochs=nb_iteration)

predict = tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,),
use_bias=True)
model = tf.keras.Sequential([predict])

print(predict.get_weights())