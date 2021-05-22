#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
t0 = time()
y_pred = gnb.fit(features_train, labels_train).predict(features_train)


print("training time:", round(time()-t0, 3), "s")


print("Number of mislabeled points out of a total %d points : %d"% (features_train.shape[0], (labels_train != y_pred).sum()))
pred_acc = (1-(((labels_train != y_pred).sum())/features_train.shape[0]))*100
print("Accuracy = {} %".format(pred_acc))
#########################################################
