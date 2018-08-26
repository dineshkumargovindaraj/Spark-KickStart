from pyspark import SparkContext
import pandas
import os
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv


sc = SparkContext.getOrCreate()
# Input File
data_file = "C:\\Users\\dkdin\\Desktop\\DataMining\\DataMining\\Spark Assignment\\flowers.txt"
rddToData = "C:\\Users\\dkdin\\Desktop\\DataMining\\DataMining\\Spark Assignment\\flowersdata.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# Txt to rdd
rdd = sc.textFile(data_file)
rddToList = rdd.collect()
print(rddToList) 
    
with open(rddToData, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for items in rddToList: wr.writerow(items.split(','))

	
dataset = pandas.read_csv(rddToData,names = names)
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#print X_validation
#print Y_validation
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
accuracy_result=[]
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    accuracy_result.append(cv_results.mean())

	
# Make predictions on validation dataset
model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

import matplotlib.pyplot as plt
algorithms = names
accuracy = accuracy_result
plt.plot(algorithms, accuracy, color='DarkBlue', label='Accuracy %')
plt.xlabel('Machine Learning Algorithms')
plt.ylabel('Accuracy of Algorithms')
plt.title('Machine learning algorithms accuracy for flowers dataset')
plt.legend(loc='upper left')
plt.show()

sc.stop()