from __future__ import print_function
from xml.sax.handler import feature_namespace_prefixes
import pandas as pd
from pdpbox import pdp, get_dataset, info_plots
import matplotlib
print(matplotlib.__version__)
import matplotlib.pyplot as plt
import sklearn 
print(sklearn.__version__)
import autosklearn.classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


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
labels = data[1:,17]
print (labels)
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[1:,2:17]
print (data)

# Data processing
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

# Automl model training
classifier = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task = max_seconds,
        per_run_time_limit = max_seconds // 10,
        metric = autosklearn.metrics.f1,
)

classifier.fit(data, labels, dataset_name='shearwall')

# data transformation for PDP analysis
features = ['Yield Stresses of Vertical Bars (MPa)', 'Yield Stresses of Horizontal Reinforcement (MPa)', 'Yield Stress of Confinement Reinforcement (MPa)', 'Concrete Compressive Strength (MPa)', 'Web Vertical Reinforcement Ratio', 'Boundary Region Vertical Reinforcement Ratio','Web Horizontal Reinforcement Ratio', 'Boundary Region (Volume) Horizontal Reinforcement Ratio','H (mm)','Ag (mm^2)','Af(mm^2)','P/fcAg','Section_R','Section_B','Section_F']
titanic_features = features
titanic_model = classifier
titanic_target = 'FailureMode'
data1 = pd.DataFrame(data, columns = ['Yield Stresses of Vertical Bars (MPa)', 'Yield Stresses of Horizontal Reinforcement (MPa)', 'Yield Stress of Confinement Reinforcement (MPa)', 'Concrete Compressive Strength (MPa)', 'Web Vertical Reinforcement Ratio', 'Boundary Region Vertical Reinforcement Ratio','Web Horizontal Reinforcement Ratio', 'Boundary Region (Volume) Horizontal Reinforcement Ratio','H (mm)','Ag (mm^2)','Af(mm^2)','P/fcAg','Section_R','Section_B','Section_F'])
data2 = pd.DataFrame(labels, columns = ['FailureMode'])
dataf=pd.concat([data1,data2],axis=1)
datas= dataf.astype(float)
titanic_data = datas
# PDP analysis for each feature
# 1.concrete----------------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Concrete Compressive Strength (MPa)', feature_name='Concrete strength', target=titanic_target, show_percentile=True
)


plt.show()
plt.savefig('result/concrete/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Concrete Compressive Strength (MPa)', feature_name='Concrete strength',
    show_percentile=True, predict_kwds={}
)


plt.show()
plt.savefig('result/concrete/picture2.svg')

# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Concrete Compressive Strength (MPa)', num_grid_points=19
)


fig, axes = pdp.pdp_plot(pdp_fare, 'Concrete strength')

fig, axes = pdp.pdp_plot(pdp_fare, 'Concrete strength', plot_pts_dist=True)


plt.show()
plt.savefig('result/concrete/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)



fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Concrete Compressive Strength (MPa)', plot_pts_dist=True
)


X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)
Z= pd.DataFrame(pdp_fare.ice_lines)

X.to_excel(r'dataresult/concrete/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/concrete/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/concrete/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/concrete/picture4.svg')


# 2.Yield Stresses of Vertical Bars (MPa)----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Yield Stresses of Vertical Bars (MPa)', feature_name='Yield Stresses of Vertical Bars', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Yield Stresses of Vertical Bars (MPa)', feature_name='Yield Stresses of Vertical Bars', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Yield Stresses of Vertical Bars (MPa)', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Vertical Bars')

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Vertical Bars', plot_pts_dist=True)

plt.show()
plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Yield Stresses of Vertical Bars (MPa)', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)


Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Yield Stresses of Vertical Bars (MPa)/Z.xlsx', sheet_name='train', index = False)

X.to_excel(r'dataresult/Yield Stresses of Vertical Bars (MPa)/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Yield Stresses of Vertical Bars (MPa)/Y.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture4.svg')

# 3.Yield Stresses of Horizontal Reinforcement (MPa)----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Yield Stresses of Horizontal Reinforcement (MPa)', feature_name='Yield Stresses of Horizontal Reinforcement', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Yield Stresses of Horizontal Reinforcement (MPa)', feature_name='Yield Stresses of Horizontal Reinforcement', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Yield Stresses of Horizontal Reinforcement (MPa)', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Horizontal Reinforcement')

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Horizontal Reinforcement', plot_pts_dist=True)

plt.show()
plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Yield Stresses of Horizontal Reinforcement (MPa)', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Yield Stresses of Horizontal Reinforcement (MPa)/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Yield Stresses of Horizontal Reinforcement (MPa)/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Yield Stresses of Horizontal Reinforcement (MPa)/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture4.svg')

# 4.Yield Stress of Confinement Reinforcement (MPa)----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Yield Stress of Confinement Reinforcement (MPa)', feature_name='Yield Stress of Confinement Reinforcement', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Yield Stress of Confinement Reinforcement (MPa)', feature_name='Yield Stress of Confinement Reinforcement', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Yield Stress of Confinement Reinforcement (MPa)', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stress of Confinement Reinforcement')

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stress of Confinement Reinforcement', plot_pts_dist=True)

plt.show()
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Yield Stress of Confinement Reinforcement (MPa)', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Yield Stress of Confinement Reinforcement (MPa)/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Yield Stress of Confinement Reinforcement (MPa)/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Yield Stress of Confinement Reinforcement (MPa)/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture4.svg')

# 5.Web Vertical Reinforcement Ratio----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Web Vertical Reinforcement Ratio', feature_name='Web Vertical Reinforcement Ratio', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Web Vertical Reinforcement Ratio/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Web Vertical Reinforcement Ratio', feature_name='Web Vertical Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Web Vertical Reinforcement Ratio/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Web Vertical Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Web Vertical Reinforcement Ratio')

fig, axes = pdp.pdp_plot(pdp_fare, 'Web Vertical Reinforcement Ratio', plot_pts_dist=True)

plt.show()
plt.savefig('result/Web Vertical Reinforcement Ratio/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Web Vertical Reinforcement Ratio', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Web Vertical Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Web Vertical Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)
Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Web Vertical Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)


plt.show()
plt.savefig('result/Web Vertical Reinforcement Ratio/picture4.svg')


# 6.Boundary Region Vertical Reinforcement Ratio----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Boundary Region Vertical Reinforcement Ratio', feature_name='Boundary Region Vertical Reinforcement Ratio', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Boundary Region Vertical Reinforcement Ratio', feature_name='Boundary Region Vertical Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Boundary Region Vertical Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Boundary Region Vertical Reinforcement Ratio')

fig, axes = pdp.pdp_plot(pdp_fare, 'Boundary Region Vertical Reinforcement Ratio', plot_pts_dist=True)

plt.show()
plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Boundary Region Vertical Reinforcement Ratio', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Boundary Region Vertical Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Boundary Region Vertical Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Boundary Region Vertical Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture4.svg')


# 7.Web Horizontal Reinforcement Ratio----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Web Horizontal Reinforcement Ratio', feature_name='Web Horizontal Reinforcement Ratio', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Web Horizontal Reinforcement Ratio/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Web Horizontal Reinforcement Ratio', feature_name='Web Horizontal Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Web Horizontal Reinforcement Ratio/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Web Horizontal Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Web Horizontal Reinforcement Ratio')

fig, axes = pdp.pdp_plot(pdp_fare, 'Web Horizontal Reinforcement Ratio', plot_pts_dist=True)

plt.show()
plt.savefig('result/Web Horizontal Reinforcement Ratio/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Web Horizontal Reinforcement Ratio', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Web Horizontal Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Web Horizontal Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Web Horizontal Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Web Horizontal Reinforcement Ratio/picture4.svg')

# 8.Boundary Region (Volume) Horizontal Reinforcement Ratio---------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Boundary Region (Volume) Horizontal Reinforcement Ratio', feature_name='Boundary Region (Volume) Horizontal Reinforcement Ratio', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Boundary Region (Volume) Horizontal Reinforcement Ratio', feature_name='Boundary Region (Volume) Horizontal Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Boundary Region (Volume) Horizontal Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Boundary Region (Volume) Horizontal Reinforcement Ratio')

fig, axes = pdp.pdp_plot(pdp_fare, 'Boundary Region (Volume) Horizontal Reinforcement Ratio', plot_pts_dist=True)

plt.show()
plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Boundary Region (Volume) Horizontal Reinforcement Ratio', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture4.svg')


# 9.lw/tw---------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='lw/tw', feature_name='lw/tw', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/lwtw/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='lw/tw', feature_name='lw/tw', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/lwtw/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='lw/tw', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'lw/tw')

fig, axes = pdp.pdp_plot(pdp_fare, 'lw/tw', plot_pts_dist=True)

plt.show()
plt.savefig('result/lwtw/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'lw/tw', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/lwtw/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/lwtw/Y.xlsx', sheet_name='train', index = False)


Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/lwtw/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/lwtw/picture4.svg')

# 10.Aspect ratio--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Aspect ratio', feature_name='Aspect ratio', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/Aspect ratio/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Aspect ratio', feature_name='Aspect ratio', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Aspect ratio/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Aspect ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Aspect ratio')

fig, axes = pdp.pdp_plot(pdp_fare, 'Aspect ratio', plot_pts_dist=True)

plt.show()
plt.savefig('result/Aspect ratio/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Aspect ratio', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Aspect ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Aspect ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Aspect ratio/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Aspect ratio/picture4.svg')

# 11.Af(mm^2)--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Af/Ag', feature_name='Af/Ag', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/AfAg/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Af/Ag', feature_name='Af/Ag', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/AfAg/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Af/Ag', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Af/Ag')

fig, axes = pdp.pdp_plot(pdp_fare, 'Af/Ag', plot_pts_dist=True)

plt.show()
plt.savefig('result/AfAg/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'Af/Ag', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/AfAg/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/AfAg/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/AfAg/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/AfAg/picture4.svg')

# 12.P/fcAg--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='P/fcAg', feature_name='P/fcAg', target=titanic_target, show_percentile=True
)

summary_df

plt.show()
plt.savefig('result/PfcAg/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='P/fcAg', feature_name='P/fcAg', 
    show_percentile=True, predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/PfcAg/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='P/fcAg', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'P/fcAg')

fig, axes = pdp.pdp_plot(pdp_fare, 'P/fcAg', plot_pts_dist=True)

plt.show()
plt.savefig('result/PfcAg/picture3.svg')
# fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)

fig, axes = pdp.pdp_plot(
    pdp_fare, 'P/fcAg', frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/PfcAg/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/PfcAg/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/PfcAg/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/PfcAg/picture4.svg')

# 12.Section--------------------------------------------------------------------------------------------------------------------------------------------

fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature=['Section_R', 'Section_B', 'Section_F'], feature_name='Section', 
    target=titanic_target
)

summary_df

plt.show()
plt.savefig('result/Section/picture1.svg')

# 2.2 check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature=['Section_R', 'Section_B', 'Section_F'], 
    feature_name='Section', predict_kwds={}
)

summary_df

plt.show()
plt.savefig('result/Section/picture2.svg')
# 2.3 pdp for feature 'embarked'

print ('1')

titanic_data= titanic_data.astype(float)

pdp_embark = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, 
    feature=['Section_R', 'Section_B', 'Section_F']
)

print ('2')

fig, axes = pdp.pdp_plot(pdp_embark, 'Section')

X= pd.DataFrame(pdp_embark.display_columns)
Y= pd.DataFrame(pdp_embark.pdp)

X.to_excel(r'dataresult/Section/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Section/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_embark.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Section/Z.xlsx', sheet_name='train', index = False)

plt.show()
plt.savefig('result/Section/picture3.svg')
fig, axes = pdp.pdp_plot(pdp_embark, 'Section', center=True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True)
