#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

#data = featureFormat(data_dict, features_list)
data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')

labels, features = targetFeatureSplit(data)



### your code goes here 
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,
                labels,test_size=0.3,random_state=42)
                
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
#acc = clf.score(features_test,labels_test)
#print(acc)
pred = clf.predict(features_test)
print (pred)
# [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.
#  0.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
# nb of poi = 4
print(len(pred))  # 29
print("labels_test = ", labels_test)
confusion = confusion_matrix(labels_test, pred)
print(confusion)

print (precision_score(labels_test, pred))

print (recall_score(labels_test, pred))



