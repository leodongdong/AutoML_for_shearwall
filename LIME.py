from __future__ import print_function
feature_names = ["fy,vw", "fy,hw", "fy,cb", "fc", "?vw", "?vb","?hw", "?hb","Section","lw/tw","Aspect ratio","Ab/Ag","P/fcAg"]
class_names=["FailureMode"]

import sklearn
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib as mpl
import autosklearn.classification
from sklearn.model_selection import KFold
np.random.seed(1)

# -*- coding=utf-8 -*-
import csv
p = r'/mnt/c/Users/user/Dropbox/Mahchine learning/Shaer Wall prediction/Analysis/Final/cate/LIME/X_train1.csv'
with open(p,encoding = 'utf-8') as f:
    data = np.loadtxt(f,str,delimiter = ",", skiprows = 1)
    print(data[:5])

labels = data[:,14]
print (labels)
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:14]
print (data)

categorical_features = [8]

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:,feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

print (data[:5])

data = data.astype(float)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [8]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)

transformer.fit(data)
print (data.shape)
encoded_train = transformer.transform(data)
print (encoded_train.shape)

print ('1')

classifier = autosklearn.classification.AutoSklearnClassifier(
  time_left_for_this_task=30,
  per_run_time_limit=None,
  tmp_folder='/tmp/autosklearn_classification_example_229',
)

classifier.fit(encoded_train, labels, dataset_name='shearwall')

print('Accuracy of automl classifier on training set: {:.2f}'
  .format(classifier.score(encoded_train, labels)))

# import joblib
# 保存模型

import pickle
with open('automl.pickle', 'wb') as handle:
    pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('automl.pickle', 'rb') as handle:
    automl = pickle.load(handle)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# -*- coding=utf-8 -*-
import csv
p = r'/mnt/c/Users/user/Dropbox/Mahchine learning/Shaer Wall prediction/Analysis/Final/cate/LIME/X_train1.csv'
with open(p,encoding = 'utf-8') as f:
    data = np.loadtxt(f,str,delimiter = ",", skiprows = 1)
    print(data[:5])

labels = data[:,14]
print (labels)
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:14]
print (data)

categorical_features = [8]

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:,feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_
print (data)

data = data.astype(float)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [8]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)

transformer.fit(data)
print (data.shape)
encoded_train = transformer.transform(data)
print (encoded_train.shape)

print ('1')

# test1 = parameter

predict_fn = lambda x: automl.predict_proba(transformer.transform(x)).astype(float)

# Explaining predictions

explainer = lime.lime_tabular.LimeTabularExplainer(data ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)

i = 1
exp = explainer.explain_instance(data[i], predict_fn, num_features=13)

print (exp)
fig = exp.as_pyplot_figure()
exp.show_in_notebook(show_table=True, show_all=False)

exp.save_to_file('lime1.html')

i = 2
exp = explainer.explain_instance(data[i], predict_fn, num_features=13)

print ('9')
# fig = exp.as_pyplot_figure()
# exp.show_in_notebook(show_table=True, show_all=False)

exp.save_to_file('lime2.html')
print (exp.random_state)
print (exp.mode)
print (exp.domain_mapper)
print (exp.local_exp)
print (exp.intercept)

print (exp.class_names)
print (exp.top_labels)
print (exp.predict_proba)


i = 110
exp = explainer.explain_instance(data[i], predict_fn, num_features=14)

print ('9')
fig = exp.as_pyplot_figure()


plt.savefig('images/plot.png',format='png')
exp.save_to_file('lime110.html')

plt.show()




