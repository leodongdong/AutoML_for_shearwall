#from __future__ import print_function
import sklearn as sk
import pandas as pd
import numpy as np
import autosklearn
import autosklearn.classification
import sklearn.preprocessing as preprocessing
import sklearn.model_selection 
import sklearn.compose

# maximum time
max_seconds = 100
# reading data from Excel
import xlrd
book = xlrd.open_workbook('Shear_Wall_dataset.xlsx') 
data = book.sheet_by_name('stage_1_pre_random')         # Pre-randomized sequence for  
rows = data.nrows  # acquire row lines
cols = data.ncols  # acquire column lines
allline = []  # save all rows
for i in range(rows):
    line = []  # save single row
    for j in range(cols):
        cell = data.cell_value(i, j)
        try:
            line.append(cell)
        except ValueError as a:
            pass
    allline.append(line)  # save single row line
data = np.array(allline)

# encoding string labels to integer
labels = data[1:,14]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
data = data[1:,1:14]

# encoding categorical_features from string to high-dim value
categorical_features = [8]
for feature in categorical_features:
    le = sk.preprocessing.LabelEncoder()
    le.fit(data[:,feature])
    data[:, feature] = le.transform(data[:, feature])

data = data.astype(float)

transformer = sk.compose.ColumnTransformer(
    transformers=[
        ("transform_CS", # Name
        preprocessing.OneHotEncoder(), # The transformer class
        categorical_features           # The column(s) to be applied on.
        )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
transformer.fit(data)

mc=0    # matrix of confusion
# loop for 10 folds of pre-randomized table
kf = sk.model_selection.KFold(n_splits=10)
kf.get_n_splits(data)
for train_index, test_index in kf.split(data):
    # collect training data
    train = data[train_index]
    labels_train = labels[train_index]
    encoded_train = transformer.transform(train)

    # AutoML
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task = max_seconds,
        per_run_time_limit = max_seconds // 10,
        metric = autosklearn.metrics.f1,
    #    tmp_folder='/tmp/autosklearn_classification_example_6',
    )
    automl.fit(encoded_train, labels_train)

    # prediction using trained AutoML
    test = transformer.transform(data[test_index])
    predictions = automl.predict(test)

    # compute confusion matrix
    labels_test = labels[test_index]
    mc += sk.metrics.confusion_matrix(labels_test, predictions)
    
# output results
print(mc, 'in', max_seconds, '* 10 seconds')
