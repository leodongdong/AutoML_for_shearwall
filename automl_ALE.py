# matplotlib inline
from __future__ import print_function
from xml.sax.handler import feature_namespace_prefixes
import pandas as pd
import matplotlib
print(matplotlib.__version__)
import matplotlib.pyplot as plt
import sklearn
print(sklearn.__version__)
import autosklearn.classification
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE, plot_ale

# maximum time
max_seconds = 100
# reading data from Excel
import xlrd
book = xlrd.open_workbook('Shear_Wall_dataset.xlsx') 
data = book.sheet_by_name('stage_1_pre_random_inter')         # Pre-randomized sequence for  
rows = data.nrows  # 获取总行数
cols = data.ncols  # 获取总列数
allline = []  # 存储所有行
for i in range(rows):
    line = []  # 存储单行数据
    for j in range(cols):
        cell = data.cell_value(i, j)
        try:
            line.append(cell)
        except ValueError as a:
            pass
    allline.append(line)  # 单行数据保存
data = np.array(allline)

feature_names = ['Yield Stresses of Vertical Bars (MPa)', 'Yield Stresses of Horizontal Reinforcement (MPa)', 'Yield Stress of Confinement Reinforcement (MPa)', 'Concrete Compressive Strength (MPa)', 'Web Vertical Reinforcement Ratio', 'Boundary Region Vertical Reinforcement Ratio','Web Horizontal Reinforcement Ratio', 'Boundary Region (Volume) Horizontal Reinforcement Ratio','lw/tw','Aspect ratio','Ab/Ag','P/fcAg','Section_R','Section_B','Section_F']
target_names = ['FailureMode']

labels = data[1:,17]
print (labels)
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[1:,2:17]
print (data)

# data processing
categorical_features = [12,13,14]
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
         [12,13,14]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
transformer.fit(data)
encoded_train = transformer.transform(data)


# AutoML training
lr = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task = max_seconds,
        per_run_time_limit = max_seconds // 10,
        metric = autosklearn.metrics.f1,
)
lr.fit(data, labels, dataset_name='shearwall')

#ALE analysis
proba_fun_lr = lr.predict_proba
proba_ale_lr = ALE(proba_fun_lr, feature_names=feature_names, target_names=target_names)
proba_exp_lr = proba_ale_lr.explain(data)
alevalues = proba_exp_lr["ale_values"]
ale0 = proba_exp_lr["ale0"]
feature_values = proba_exp_lr["feature_values"]
alevalues= pd.DataFrame(alevalues)
ale0= pd.DataFrame(ale0)
feature_values= pd.DataFrame(feature_values)

# ALE analysis result save
alevalues.to_excel(r'alevalues1.xlsx', sheet_name='train', index = False)
ale0.to_excel(r'ale01.xlsx', sheet_name='train', index = False)
feature_values.to_excel(r'feature_values1.xlsx', sheet_name='train', index = False)
plot_ale(proba_exp_lr, n_cols=2, fig_kw={'figwidth': 80, 'figheight': 50}, sharey=None)
plt.savefig('picture1.jpg')
