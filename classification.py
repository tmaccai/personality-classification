#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:38:18 2018

@author: huantan
"""
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.neighbors import KNeighborsClassifier  
import sys

filename = sys.argv[1]+'.p'
x = pickle.load(open(filename, 'rb'))
revs, vocab, mairesse, uni, bi, tri, train_id, test_id, nn_pred = x[0], x[1], x[2], x[3], x[4],x[5],x[6],x[7],x[8]
# tf-idf
tfidf = pickle.load(open('tfidf.p', 'rb'))

train_x = np.empty((0,600))
mairess_trainx = np.empty((0,684))
tfidf_trainx, mt_trainx = np.empty((0,800)), np.empty((0,884))
train_y = []

for i in train_id:
    name = revs[i]['user']
    a = uni[name]+bi[name]+tri[name]
    train_x = np.append(train_x,np.array(a).reshape(1,600), axis = 0)
    mairess_trainx = np.append(mairess_trainx,np.array(a+mairesse[name]).reshape(1,684), axis = 0)
    tfidf_trainx = np.append(tfidf_trainx, np.array(a + list(tfidf[name])).reshape(1, 800), axis=0)
    mt_trainx = np.append(mt_trainx, np.array(a + mairesse[name]+ list(tfidf[name])).reshape(1, 884), axis=0)
    train_y.append(revs[i]['y0'])
    
    
test_x = np.empty((0,600))
mairess_testx = np.empty((0,684))
tfidf_testx, mt_testx = np.empty((0,800)), np.empty((0,884))
test_y = []

for i in test_id:
    name = revs[i]['user']
    a = uni[name]+bi[name]+tri[name]
    test_x = np.append(test_x,np.array(a).reshape(1,600), axis = 0)
    mairess_testx = np.append(mairess_testx,np.array(a+mairesse[name]).reshape(1,684), axis = 0)
    tfidf_testx = np.append(tfidf_testx, np.array(a + list(tfidf[name])).reshape(1, 800), axis=0)
    mt_testx = np.append(mt_testx, np.array(a + mairesse[name] + list(tfidf[name])).reshape(1, 884),axis=0)
    test_y.append(revs[i]['y0'])



## svm
clf = Pipeline([('clf',svm.SVC(C=1,kernel='linear',probability=True))
                    #                       SGDClassifier(loss='modified_huber', penalty='l2',
#                                           alpha=1e-3, random_state=42,
#                                           max_iter=5, tol=None)),
                    ])

        
#text_clf.fit(x_train, x_score)

parameters = {'clf__C': (0.001, 0.01, 0.1, 1,2),
              #'clf__alpha': (1e-2, 1e-3,1e-4),
             }

###  svm train
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train_x, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
predicted=gs_clf.predict(train_x) 
print('svm train',np.mean(predicted == train_y)) 

predicted=gs_clf.predict(test_x) 
print('svm test',np.mean(predicted == test_y))

### svm mairess train
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(mairess_trainx, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
predicted=gs_clf.predict(mairess_trainx) 
print('svm mairess train',np.mean(predicted == train_y)) 

predicted=gs_clf.predict(mairess_testx) 
print('svm mairess test',np.mean(predicted == test_y))

### svm tfidf
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(tfidf_trainx, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

predicted = gs_clf.predict(tfidf_trainx)
print('svm tfidf train', np.mean(predicted == train_y))

predicted = gs_clf.predict(tfidf_testx)
print('svm tfidf test', np.mean(predicted == test_y))

### svm mairesse and tfidf train
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(mt_trainx, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

predicted = gs_clf.predict(mt_trainx)
print('svm mairess tfidf train', np.mean(predicted == train_y))

predicted = gs_clf.predict(mt_testx)
print('svm mairess tfidf test', np.mean(predicted == test_y))

################################################

## logistic regression
clf = Pipeline([('lg',LogisticRegression())])
parameters = {'lg__penalty': [ "l1", "l2"],}

gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)


### logistic  train
gs_clf = gs_clf.fit(train_x, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
predicted=gs_clf.predict(train_x) 
print('logistic train',np.mean(predicted == train_y))

predicted=gs_clf.predict(test_x) 
print('logistic test',np.mean(predicted == test_y))

### logistic mairess train
gs_clf = gs_clf.fit(mairess_trainx, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
predicted=gs_clf.predict(mairess_trainx) 
print('logistic mairess train',np.mean(predicted == train_y))

predicted=gs_clf.predict(mairess_testx) 
print('logistic mairess test',np.mean(predicted == test_y))

### logistic tfidf train
gs_clf = gs_clf.fit(tfidf_trainx, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

predicted = gs_clf.predict(tfidf_trainx)
print('logistic tfidf train', np.mean(predicted == train_y))

predicted = gs_clf.predict(tfidf_testx)
print('logistic tfidf test', np.mean(predicted == test_y))

### logistic mt train
gs_clf = gs_clf.fit(mt_trainx, train_y)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

predicted = gs_clf.predict(mt_trainx)
print('logistic mairess tfidf train', np.mean(predicted == train_y))

predicted = gs_clf.predict(mt_testx)
print('logistic mairess tfidf test', np.mean(predicted == test_y))

##################################################
## neural networks

nn_trainy = np.vstack((np.array(train_y), 1-np.array(train_y))).T
nn_testy = np.vstack((np.array(test_y), 1-np.array(test_y))).T


### nn train
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=600))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_x,nn_trainy, epochs=1, batch_size=1)

score, acc = model.evaluate(train_x, nn_trainy,
                            batch_size=1)
print('neural network train', acc)
score, acc = model.evaluate(test_x, nn_testy,
                            batch_size=1)
print('neural network test', acc)

### nn mairresse train
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=684))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(mairess_trainx,nn_trainy, epochs=1, batch_size=1)

score, acc = model.evaluate(mairess_trainx, nn_trainy,
                            batch_size=1)
print('neural network mairess train', acc)
score, acc = model.evaluate(mairess_testx, nn_testy,
                            batch_size=1)
print('neural network mairess test', acc)

### nn mairresse train
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=800))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(tfidf_trainx,nn_trainy, epochs=1, batch_size=1)

score, acc = model.evaluate(tfidf_trainx, nn_trainy,
                            batch_size=1)
print('neural network tfidf train', acc)
score, acc = model.evaluate(tfidf_testx, nn_testy,
                            batch_size=1)
print('neural network tfidf test', acc)

### nn mairresse train
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=884))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(mt_trainx,nn_trainy, epochs=1, batch_size=1)

score, acc = model.evaluate(mt_trainx, nn_trainy,
                            batch_size=1)
print('neural network mairess tfidf train', acc)
score, acc = model.evaluate(mt_testx, nn_testy,
                            batch_size=1)
print('neural network mairess tfidf test', acc)

###############################################
# knn train
  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(train_x, train_y)
predicted = classifier.predict(train_x)
print('knn train',np.mean(predicted == train_y))
predicted = classifier.predict(test_x)
print('knn test',np.mean(predicted == test_y))


# knn mairesse train

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(mairess_trainx, train_y)
predicted = classifier.predict(mairess_trainx)
print('knn mairesse train',np.mean(predicted == train_y))
predicted = classifier.predict(mairess_testx)
print('knn mairesse test',np.mean(predicted == test_y))


# knn tfidf train
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(tfidf_trainx, train_y)
predicted = classifier.predict(tfidf_trainx)
print('knn tfidf train',np.mean(predicted == train_y))
predicted = classifier.predict(tfidf_testx)
print('knn tfidf train',np.mean(predicted == test_y))


# knn mairesse tfidf train
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(mt_trainx, train_y)
predicted = classifier.predict(mt_trainx)
print('knn mairesse tfidf train',np.mean(predicted == train_y))
predicted = classifier.predict(mt_testx)
print('knn mairesse tfidf train',np.mean(predicted == test_y))

























