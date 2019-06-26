#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:29:42 2019

@author: nicolas
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import seaborn
import pandas
import matplotlib.pyplot as plt

"""CONSTRUCTION DU GRAPHE"""
w = tf.Variable(tf.zeros((2, 1))) #w: initialisée par 0
b = tf.Variable(-0.5, tf.float32) #b: -0.5
x = tf.placeholder(tf.float32, (4, 2)) # entrée
predict = tf.nn.sigmoid(tf.matmul(x, w) + b) # la sortie (prediction) 


#FONCTION DE PERTE
    
y = tf.placeholder(tf.float32, (4, 1)) #sortie souhaitée
loss = tf.reduce_sum(tf.square(y - predict), 0) #fonction de perte
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss) #optimiseur

#ENTRAINEMENT
data_x = np.array([[0, 0],[0, 1],[1, 0],[1, 1]]) #donnée d'apprentissage: x
data_y = np.array([[0.0], [1.0], [1.0], [1.0]]) #donnée d'apprentissage: y

nb_it = 4000 #nombre d’itérations
sess = tf.Session()

sess.run(tf.global_variables_initializer()) #init des variables
sess.run(w.assign([[0.4], [0.7]])) #initialiser w = [0.4, 0.5]

for i in range(0, nb_it):
    sess.run(train_step, {x: data_x, y: data_y}) #entrainer le perceptron

#PREDICTION
p = sess.run(predict, {x: data_x}) #prédire sur l’ensemble d’apprentissage
w_value = sess.run(w) #prendre les valeurs de w et
b_value = sess.run(b)
print("prediction = ",(p>0.5).astype(int))
print("W1 = ", w_value[0][0])
print("W2 = ", w_value[1][0])
print("b = ",b_value)

dim1 = [0, 0, 1, 1]
dim2 = [0, 1, 0, 1]
data_y = [0, 1, 1, 1]

dataFrame = pandas.DataFrame({"x1": dim1,
                              "x2": dim2,
                               "Classe":data_y})

trace = seaborn.lmplot("x1", "x2", hue = "Classe", data = dataFrame, fit_reg = False)



x1 = np.arange(0, 1, 0.2)
x2 = (1-b_value - w_value[0][0]*x1)/w_value[1][0]
x3 = (1-b_value - w_value[1][0]*x1)/w_value[0][0]
x4 = (-1*b_value - w_value[1][0]*x1)/w_value[0][0]
y1 = -1*x1 +1
y2 = -1*x1
plt.plot(x1, y2, 'gray',linestyle='dashed')
plt.plot(x1, y1, 'black', linestyle='dashed')
plt.plot(x1, x3, 'red')
plt.plot(x1, x2, 'blue')
plt.plot(x1, x4, 'red')
plt.show()