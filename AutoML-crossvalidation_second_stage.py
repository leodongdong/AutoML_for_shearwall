from __future__ import print_function
feature_names = ["Yield Stresses of Vertical Bars (MPa)", "Yield Stresses of Horizontal Reinforcement (MPa)", "Yield Stress of Confinement Reinforcement (MPa)", "Concrete Compressive Strength (MPa)", "Web Vertical Reinforcement Ratio", "Boundary Region Vertical Reinforcement Ratio","Web Horizontal Reinforcement Ratio", "Boundary Region (Volume) Horizontal Reinforcement Ratio","Section","lw/tw","Aspect ratio","Af/Ag","P/fcAg"]
class_names=["FailureMode"]

import sklearn
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import autosklearn.classification
from sklearn.model_selection import KFold
from pprint import pprint
np.random.seed(1)

dict= [50]
# dict= [30]

maxyx = 0 

for time1 in dict:
    # -*- coding=utf-8 -*-
    import csv
    p = r'/home//leo/leo/MLtest/X_train1.csv'
    with open(p,encoding = 'utf-8') as f:
        data = np.loadtxt(f,str,delimiter = ",", skiprows = 1)
        print(data[:5])

    labels = data[:,14]
    le= sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    class_names = le.classes_
    data = data[:,1:14]


    kf = KFold(n_splits=10)
    kf.get_n_splits(data)

    categorical_features = [8]

    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:,feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_


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

    KFold(n_splits=10, random_state=None, shuffle=False)
    i=1
    ac=0
    mc=0
    for train_index, test_index in kf.split(data):
        train = data[train_index]
        labels_train = labels[train_index]
        test = data[test_index]
        labels_test = labels[test_index]

        transformer.fit(data)
        encoded_train = transformer.transform(train)

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time1,
            per_run_time_limit=None,
            tmp_folder='/tmp/autosklearn_classification_example_2984',
        )

        automl.fit(encoded_train, labels_train, dataset_name='shearwall')
        pprint(automl.show_models(), indent=4)
        print(automl.leaderboard())
        print('hello world')
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        
        test = test.astype(float)
        test1 = transformer.transform(test)
        
        predictions = automl.predict(test1)
        ac=ac+sklearn.metrics.accuracy_score(labels_test, predictions)

        confusion_matrix = confusion_matrix(labels_test, predictions)
        mc=mc+confusion_matrix
        i=i+1

    yx=ac/10
    print(time1)
    print(yx)
    print(mc)






