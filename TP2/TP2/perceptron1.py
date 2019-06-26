#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:29:48 2019

@author: nicolas
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

w = tf.Variable(tf.random_uniform((2, 3), -1, -1)) #w: initialisée aleatoirement
b = tf.Variable(tf.zeros((3,)), tf.float32) #b:
x = tf.placeholder(tf.float32, (None, 2)) # entrée
predict = tf.nn.softmax(tf.matmul(x, w) + b) # la sortie (prediction)

data_x = np.array([[1, 0, 0],[0, 1,0],[0, 0,1 ]]) #donnée d'apprentissage: x


label = data_x[:, -1].astype(int)
(m, n) = data_x.shape
#coder les classes
C = 3
data_y = [[0 for j in range(C)] for i in range(m)]
for i in range(m):
    data_y[i][label[i]] = 1

y = tf.placeholder(tf.float32, (None, 3)) #sortie souhaitée
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(predict), 1)) #fonction de perte
train_step = tf.train.AdamOptimizer(0.01).minimize(loss) #optimiseur

nb_it = 100
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #init des variables
sess.run(w.assign([[0.4], [0.5]])) #initialiser w = [0.4, 0.5]

for i in range(0, nb_it):
    sess.run(train_step, {x: data_x, y: data_y}) #entrainer le perceptron
    
p = sess.run(predict, {x: data_x}) #prédire
print(np.argmax(p[5])) #la classe prédicte de la 5e indiviu
print(label[5]) #la classe actuelle
print("Accuracy =", sum(np.argmax(p, axis=1) == label)/m)