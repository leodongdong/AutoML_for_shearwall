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

feature_names = ["fy,vw", "fy,hw", "fy,cb", "fc", "ρvw", "ρvb","ρhw", "ρhb","CS","lw/tw","M/Vlw","Ab/Ag","P/fcAg"]

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
class_names = le.classes_
data = data[1:,1:14]

# encoding categorical_features from string to high-dim value
categorical_features = [8]
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:,feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

data = data.astype(float)
transformer = sklearn.compose.ColumnTransformer(
    transformers=[
        ("transform_CS", # Name
        sklearn.preprocessing.OneHotEncoder(), # The transformer class
        categorical_features           # The column(s) to be applied on.
        )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
transformer.fit(data)
encoded_train = transformer.transform(data)

# automl training model
automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task = max_seconds,
        per_run_time_limit = max_seconds // 10,
        metric = autosklearn.metrics.f1,
)
automl.fit(encoded_train, labels, dataset_name='shearwall')
predict_fn = lambda x: automl.predict_proba(transformer.transform(x)).astype(float)

# Explaining predictions
explainer = lime.lime_tabular.LimeTabularExplainer(data ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)
# LIME interpretability

i = 8
print(data[i])
exp = explainer.explain_instance(data[i], predict_fn, num_features=13)
fig = exp.as_pyplot_figure()
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('lime1.html')
plt.show()

i = 29
print(data[i])
exp = explainer.explain_instance(data[i], predict_fn, num_features=13)
fig = exp.as_pyplot_figure()
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('lime2.html')
plt.show()




