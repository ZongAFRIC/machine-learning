#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:12:21 2019

@author: nicolas
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


#Creation du fichier pour stockage des descripteurs pickle
def pickle_hist(fichier,dataFrame):   
    pkl=pickle.Pickler(fichier)
    pkl.dump(dataFrame)

#Récupération du descripteur Unpickle
def unpickle_hist(fichier):   
    Unpkl=pickle.Unpickler(fichier)
    fic=(Unpkl.load())
    return fic

print("========>   EXERCICE 2 AVEC LE DATASET SPAM   <=======")
lien = "./spam/spam.data" #input("Entrer le lien du dataset: ")
nb_it = int(input("Entrer le nombre d'itération:")) #nombre d’itérations
data = pd.read_csv(lien, header = None , sep =" ")


def normalisationDataset():
    del data[58]
    for i in range(int(len(data))): 
        if int(data[57][i]) == -1.0:
            data[57][i] = 0
    print(data[57])
    #Création du fichier de stockage des dataframes
    f = open(("./dataFrameSpam"+".txt"),'wb')
    pickle_hist(f,data)
    f.close

f = open(("./dataFrameSpam"+".txt"),"rb")
data = unpickle_hist(f)
f.close


"""CONSTRUCTION DU GRAPHE"""
w = tf.Variable(tf.zeros((57, 1))) #w: initialisée par 0
b = tf.Variable(0.5, tf.float32) #b: -0.5
x = tf.placeholder(tf.float32, (100, 57)) # entrée
predict = tf.nn.sigmoid(tf.matmul(x, w) + b) # la sortie (prediction) 


#FONCTION DE PERTE  
y = tf.placeholder(tf.float32, (100, )) #sortie souhaitée
loss = tf.reduce_sum(tf.square(y - predict), 0) #fonction de perte
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss) #optimiseur

#ENTRAINEMENT
data_x = data.copy(deep = True )
taille_test = int((len(data_x)*2)/3)
del data_x[57]


data_x = data_x.iloc[:100]#donnée d'apprentissage: x
data_y = data[57][:100] #donnée d'apprentissage: y

data_xtest = data_x.iloc[:50]#donnée test: x
data_ytest = data[57][:50] #donnée test: y

sess = tf.Session()

sess.run(tf.global_variables_initializer()) #init des variables
sess.run(w) #initialiser w = [0.4, 0.5]

for i in range(0, nb_it):
    sess.run(train_step, {x: data_x, y: data_y}) #entrainer le perceptron

#PREDICTION
p = sess.run(predict, {x: data_x}) #prédire sur l’ensemble d’apprentissage
w_value = sess.run(w) #prendre les valeurs de w et
b_value = sess.run(b)
print("prediction = ",(p).astype(float))

print("W = ", w_value)

print("b = ",b_value)

