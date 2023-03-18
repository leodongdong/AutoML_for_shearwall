import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import autosklearn.classification
import numpy as np
from alibi.explainers import ALE, plot_ale

# maximum time
max_seconds = 100
# reading data from Excel
import xlrd
book = xlrd.open_workbook('Shear_Wall_dataset.xlsx') 
data = book.sheet_by_name('stage_1_pre_random_inter')         # Pre-randomized sequence for  
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


labels = data[1:,17]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[1:,2:17]


# data processing
## encoding categorical_features from string to high-dim value
categorical_features = [12,13,14]
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:,feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_
data = data.astype(float)

transformer = sklearn.compose.ColumnTransformer(
    transformers=[
        ("transform_cross section",        # Just a name
         sklearn.preprocessing.OneHotEncoder(), # The transformer class
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
feature_names = ['Yield Stresses of Vertical Bars (MPa)', 'Yield Stresses of Horizontal Reinforcement (MPa)', 'Yield Stress of Confinement Reinforcement (MPa)', 'Concrete Compressive Strength (MPa)', 'Web Vertical Reinforcement Ratio', 'Boundary Region Vertical Reinforcement Ratio','Web Horizontal Reinforcement Ratio', 'Boundary Region (Volume) Horizontal Reinforcement Ratio','lw/tw','Aspect ratio','Ab/Ag','P/fcAg','Section_R','Section_B','Section_F']
target_names = ['FailureMode']
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