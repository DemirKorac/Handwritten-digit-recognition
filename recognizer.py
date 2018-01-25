# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:10:39 2018

@author: Demir Korac
"""


import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
number = int(input('Enter the order number of the number you want from the MNIST database: '))

#importing the dataset as a matrix (in the database it is given as cell inputs of the intensity values of pixels of the 28x28 images)
data=pd.read_csv("train.csv").as_matrix()

#creating an empty classifier
clf=DecisionTreeClassifier()

#Since the dataset contains 42 000 rows and 785 columns we will take the first 21 000 rows for the training set
#starting from 0 and taking 21000 rows without the first column because it contains the order numbers 
traindata=data[0:21000,1:]

#Here we are taking the first column because we need it for the data labels
train_label=data[0:21000,0]

#Here we are training the classifier using the fit method in which it takes the coresponding data and its labels
clf.fit(traindata,train_label)

#Here we are taking the remaining part of the dataset to use as test data and taking only the first column
testdata=data [21000:,1:]

#Here we are taking the actual label of the remainder of the dataset to compare to our programs predictions 
#We are taking only the label column
actual_label=data[21000:,0]

#Here we are taking a sample from the dataset
d=testdata[number]

#We are reshaping the data from a row vector into a 28x28 matrix
d.shape=(28,28)

#Using pt as an object from matplotlib.pyplot we define the showing area
#We are writing 255-d so we get a white background and black color
pt.imshow(255-d,cmap='gray')

print ("The image you chose: ")

#Showing the image chosen from the dataset
pt.show()

#Outputing the prediction
print ("The predicted digit is =", clf.predict( [testdata[number]]))
'''
uncomment part of the code given below to see the program accuracy on the dataset
if you want to see it being calculated, insert an indentation in front of the 
print line so it lines up with the for loop

CODE:


p=clf.predict(testdata)
count=0
for i in range (0,21000):
    count+=1 if p[i]==actual_label[i] else 0
print ("Accuracy=", (count/21000)*100)

'''
