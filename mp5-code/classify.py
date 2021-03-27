# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
import heapq

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters

    W = np.zeros(shape=(np.shape(train_set)[1] + 1))
    biased_set = np.zeros(shape=(len(train_set), len(train_set[0]) + 1))

    #inserting extra feature at the start of each image array to calculate b
    for i in range(len(train_set)):
        biased_set[i] = np.insert(train_set[i], 0, 1)

    for i in range(max_iter):
        W = singleEpoch(biased_set, train_labels, learning_rate, W)

    #extracting the bias term from W and then deleting it
    return W[1:], W[0]

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set

    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    dev_labels = []

    for i in range(len(dev_set)):
        y_predict = predict(dev_set[i], W, b)

        if y_predict > 0:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here

    dev_labels = []
    for i in range(len(dev_set)):
        diff = []
        k_labels = []

        for j in range(len(train_set)):
            this_diff = np.linalg.norm(dev_set[i] - train_set[j])
            heapq.heappush(diff, (this_diff, train_labels[j]))

        for i in range(k):
            if heapq.heappop(diff)[1]:
                k_labels.append(1)
            else:
                k_labels.append(0)

        mean = np.mean(k_labels)

        if mean > 0.5:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels

def singleEpoch(train_set, train_labels, learning_rate, W):

    for i in range(len(train_set)):
        y_hat = calc_activation(train_set[i], W)
        if y_hat <= 0:
            y_hat = 0
        W += learning_rate * (train_labels[i] - y_hat) * train_set[i]

    return W

def calc_activation(current_set, W):
    F = np.dot(current_set, W)
    return np.sign(F)


def predict(current_set, W, b):
    F = np.dot(current_set, W) + b
    return np.sign(F)