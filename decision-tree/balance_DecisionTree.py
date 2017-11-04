""" ------------------------ DECISION TREES ------------------------"""
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
# Read the dtaset into pandas dataframe
balance_data = pd.read_csv('C:/Users/ANJALI/Desktop/balance-scale.data.csv',sep= ',', header= None)
# Separating the features vector and the traget variables
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]
# Splitting the datasets into train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
#Decision trees with gini criteria 
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=10, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
#Decision trees with entropy criteria 
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
#Cross validation for gini classifier
predicted = cross_val_predict(clf_gini, X_train, y_train, cv=20)
print(metrics.accuracy_score(y_train, predicted))
#Cross validation for Entropy classifier 
predicted = cross_val_predict(clf_entropy, X_train, y_train, cv=20)
print(metrics.accuracy_score(y_train, predicted)) 
#predicted score for gini classifier
y_pred = clf_gini.predict(X_test)
print( accuracy_score(y_test,y_pred)*100)
#predicted score for entropy classifier
y_pred = clf_entropy.predict(X_test)
print( accuracy_score(y_test,y_pred)*100)
# Generating visualisation diagram file
with open("gini_classifier.txt", "w") as f:
    f = tree.export_graphviz(clf_gini, out_file=f)
"""
Copy the content of the gini_classifier.txt file
go to the url - "http://webgraphviz.com/" and paste the same there to view the Visualisation diagram of the gini classifier
"""
# Generating visualisation diagram file
with open("entropy_classifier.txt", "w") as f:
    f = tree.export_graphviz(clf_entropy, out_file=f)
"""
Copy the content of the entropy_classifier.txt file
go to the url - "http://webgraphviz.com/" and paste the same there to view the Visualisation diagram of the entropy classifier
"""