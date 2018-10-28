import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import pre_processing as pp
 # Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

#Load the dataset

fileName="trainingSetLabels.csv"
train_labels = pd.read_csv(fileName)    

#Getting the pre processed data
train_variables = pp.getProcessedData("trainingSetValues.csv")

data_merged = pd.merge(train_variables,train_labels,on = 'id')

#Merging both dataframes

#Define Labels
data_merged.loc[data_merged['status_group'] == 'functional', 'dependentVariable'] = 1
data_merged.loc[data_merged['status_group'] == 'non functional', 'dependentVariable'] = -1
data_merged.loc[data_merged['status_group'] == 'functional needs repair', 'dependentVariable'] = 0    
   

nameOfTargetVariable = 'dependentVariable'

myColumns = list(train_variables)
myColumns.remove('id')

predictors = data_merged[myColumns]

targets = data_merged.dependentVariable


#60% train, 40% test
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=15)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

#confusion matrix
print (sklearn.metrics.confusion_matrix(tar_test,predictions))
print (sklearn.metrics.accuracy_score(tar_test, predictions))


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)


"""
Running a different number of trees and see the effect
on the accuracy of the prediction
"""

trees=range(15)
accuracy=np.zeros(15)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   


