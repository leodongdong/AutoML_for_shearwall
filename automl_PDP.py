import pandas as pd
from pdpbox import pdp, get_dataset, info_plots
import matplotlib.pyplot as plt
import sklearn 
import autosklearn.classification
import numpy as np
import xlrd
import os
# maximum time
max_seconds = 100
# reading data from Excel
book = xlrd.open_workbook('Shear_Wall_dataset.xlsx') 
data = book.sheet_by_name('stage_1_pre_random_inter')         # Pre-randomized sequence  
rows = data.nrows  #  acquire row lines
cols = data.ncols  # acquire column lines
allline = []  # save all rows
for i in range(rows):
    line = []  #save single row
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

# Data processing
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

# Automl model training
classifier = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task = max_seconds,
        per_run_time_limit = max_seconds // 10,
        metric = autosklearn.metrics.f1,
)

classifier.fit(data, labels, dataset_name='shearwall')

# data transformation for PDP analysis
features = ['Yield Stresses of Vertical Bars (MPa)', 'Yield Stresses of Horizontal Reinforcement (MPa)', 'Yield Stress of Confinement Reinforcement (MPa)', 'Concrete Compressive Strength (MPa)', 'Web Vertical Reinforcement Ratio', 'Boundary Region Vertical Reinforcement Ratio','Web Horizontal Reinforcement Ratio', 'Boundary Region (Volume) Horizontal Reinforcement Ratio','lw/tw','Aspect ratio','Af/Ag','P/fcAg','Section_R','Section_B','Section_F']
titanic_features = features
titanic_model = classifier
titanic_target = 'FailureMode'
data1 = pd.DataFrame(data, columns = ['Yield Stresses of Vertical Bars (MPa)', 'Yield Stresses of Horizontal Reinforcement (MPa)', 'Yield Stress of Confinement Reinforcement (MPa)', 'Concrete Compressive Strength (MPa)', 'Web Vertical Reinforcement Ratio', 'Boundary Region Vertical Reinforcement Ratio','Web Horizontal Reinforcement Ratio', 'Boundary Region (Volume) Horizontal Reinforcement Ratio','lw/tw','Aspect ratio','Af/Ag','P/fcAg','Section_R','Section_B','Section_F'])
data2 = pd.DataFrame(labels, columns = ['FailureMode'])
dataf=pd.concat([data1,data2],axis=1)
datas= dataf.astype(float)
titanic_data = datas


# PDP analysis for each feature
# 1.concrete----------------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Concrete Compressive Strength (MPa)', feature_name='Concrete strength', target=titanic_target, show_percentile=True
)
os.makedirs('result/concrete')
os.makedirs('dataresult/concrete')
plt.savefig('result/concrete/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Concrete Compressive Strength (MPa)', feature_name='Concrete strength',
    show_percentile=True, predict_kwds={}
)
plt.savefig('result/concrete/picture2.svg')

# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Concrete Compressive Strength (MPa)', num_grid_points=19
)
fig, axes = pdp.pdp_plot(pdp_fare, 'Concrete strength', plot_pts_dist=True)
plt.savefig('result/concrete/picture3.svg')


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

plt.savefig('result/concrete/picture4.svg')


# 2.Yield Stresses of Vertical Bars (MPa)----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Yield Stresses of Vertical Bars (MPa)', feature_name='Yield Stresses of Vertical Bars', target=titanic_target, show_percentile=True
)

os.makedirs('result/Yield Stresses of Vertical Bars (MPa)')
os.makedirs('dataresult/Yield Stresses of Vertical Bars (MPa)')
plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture1.svg')


fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Yield Stresses of Vertical Bars (MPa)', feature_name='Yield Stresses of Vertical Bars', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Yield Stresses of Vertical Bars (MPa)', num_grid_points=19
)
fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Vertical Bars', plot_pts_dist=True)

plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture3.svg')

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Yield Stresses of Vertical Bars (MPa)', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)
Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Yield Stresses of Vertical Bars (MPa)/Z.xlsx', sheet_name='train', index = False)
X.to_excel(r'dataresult/Yield Stresses of Vertical Bars (MPa)/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Yield Stresses of Vertical Bars (MPa)/Y.xlsx', sheet_name='train', index = False)

plt.savefig('result/Yield Stresses of Vertical Bars (MPa)/picture4.svg')

# 3.Yield Stresses of Horizontal Reinforcement (MPa)----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Yield Stresses of Horizontal Reinforcement (MPa)', feature_name='Yield Stresses of Horizontal Reinforcement', target=titanic_target, show_percentile=True
)

os.makedirs('result/Yield Stresses of Horizontal Reinforcement (MPa)')
os.makedirs('dataresult/Yield Stresses of Horizontal Reinforcement (MPa)')
plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture1.svg')


fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Yield Stresses of Horizontal Reinforcement (MPa)', feature_name='Yield Stresses of Horizontal Reinforcement', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Yield Stresses of Horizontal Reinforcement (MPa)', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Horizontal Reinforcement')
fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stresses of Horizontal Reinforcement', plot_pts_dist=True)

plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture3.svg')


fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Yield Stresses of Horizontal Reinforcement', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)
X.to_excel(r'dataresult/Yield Stresses of Horizontal Reinforcement (MPa)/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Yield Stresses of Horizontal Reinforcement (MPa)/Y.xlsx', sheet_name='train', index = False)
Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Yield Stresses of Horizontal Reinforcement (MPa)/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/Yield Stresses of Horizontal Reinforcement (MPa)/picture4.svg')

# 4.Yield Stress of Confinement Reinforcement (MPa)----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Yield Stress of Confinement Reinforcement (MPa)', feature_name='Yield Stress of Confinement Reinforcement', target=titanic_target, show_percentile=True
)

os.makedirs('result/Yield Stress of Confinement Reinforcement (MPa)')
os.makedirs('dataresult/Yield Stress of Confinement Reinforcement (MPa)')
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Yield Stress of Confinement Reinforcement (MPa)', feature_name='Yield Stress of Confinement Reinforcement', 
    show_percentile=True, predict_kwds={}
)
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Yield Stress of Confinement Reinforcement (MPa)', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Yield Stress of Confinement Reinforcement', plot_pts_dist=True)
plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture3.svg')

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Yield Stress of Confinement Reinforcement (MPa)', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)
X.to_excel(r'dataresult/Yield Stress of Confinement Reinforcement (MPa)/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Yield Stress of Confinement Reinforcement (MPa)/Y.xlsx', sheet_name='train', index = False)
Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Yield Stress of Confinement Reinforcement (MPa)/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/Yield Stress of Confinement Reinforcement (MPa)/picture4.svg')

# 5.Web Vertical Reinforcement Ratio----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Web Vertical Reinforcement Ratio', feature_name='Web Vertical Reinforcement Ratio', target=titanic_target, show_percentile=True
)

os.makedirs('result/Web Vertical Reinforcement Ratio')
os.makedirs('dataresult/Web Vertical Reinforcement Ratio')
plt.savefig('result/Web Vertical Reinforcement Ratio/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Web Vertical Reinforcement Ratio', feature_name='Web Vertical Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Web Vertical Reinforcement Ratio/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Web Vertical Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Web Vertical Reinforcement Ratio', plot_pts_dist=True)

plt.savefig('result/Web Vertical Reinforcement Ratio/picture3.svg')

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Web Vertical Reinforcement Ratio', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Web Vertical Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Web Vertical Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)
Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Web Vertical Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)
plt.savefig('result/Web Vertical Reinforcement Ratio/picture4.svg')

# 6.Boundary Region Vertical Reinforcement Ratio----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Boundary Region Vertical Reinforcement Ratio', feature_name='Boundary Region Vertical Reinforcement Ratio', target=titanic_target, show_percentile=True
)

os.makedirs('result/Boundary Region Vertical Reinforcement Ratio')
os.makedirs('dataresult/Boundary Region Vertical Reinforcement Ratio')
plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Boundary Region Vertical Reinforcement Ratio', feature_name='Boundary Region Vertical Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Boundary Region Vertical Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Boundary Region Vertical Reinforcement Ratio', plot_pts_dist=True)

plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture3.svg')

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Boundary Region Vertical Reinforcement Ratio', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Boundary Region Vertical Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Boundary Region Vertical Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Boundary Region Vertical Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/Boundary Region Vertical Reinforcement Ratio/picture4.svg')


# 7.Web Horizontal Reinforcement Ratio----------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Web Horizontal Reinforcement Ratio', feature_name='Web Horizontal Reinforcement Ratio', target=titanic_target, show_percentile=True
)

os.makedirs('result/Web Horizontal Reinforcement Ratio')
os.makedirs('dataresult/Web Horizontal Reinforcement Ratio')
plt.savefig('result/Web Horizontal Reinforcement Ratio/picture1.svg')


fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Web Horizontal Reinforcement Ratio', feature_name='Web Horizontal Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Web Horizontal Reinforcement Ratio/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Web Horizontal Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Web Horizontal Reinforcement Ratio', plot_pts_dist=True)

plt.savefig('result/Web Horizontal Reinforcement Ratio/picture3.svg')

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Web Horizontal Reinforcement Ratio', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Web Horizontal Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Web Horizontal Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Web Horizontal Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/Web Horizontal Reinforcement Ratio/picture4.svg')

# 8.Boundary Region (Volume) Horizontal Reinforcement Ratio---------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Boundary Region (Volume) Horizontal Reinforcement Ratio', feature_name='Boundary Region (Volume) Horizontal Reinforcement Ratio', target=titanic_target, show_percentile=True
)

os.makedirs('result/Boundary Region (Volume) Horizontal Reinforcement Ratio')
os.makedirs('dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio')
plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Boundary Region (Volume) Horizontal Reinforcement Ratio', feature_name='Boundary Region (Volume) Horizontal Reinforcement Ratio', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Boundary Region (Volume) Horizontal Reinforcement Ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Boundary Region (Volume) Horizontal Reinforcement Ratio', plot_pts_dist=True)

plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture3.svg')

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Boundary Region (Volume) Horizontal Reinforcement Ratio', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Boundary Region (Volume) Horizontal Reinforcement Ratio/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/Boundary Region (Volume) Horizontal Reinforcement Ratio/picture4.svg')


# 9.lw/tw---------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='lw/tw', feature_name='lw/tw', target=titanic_target, show_percentile=True
)

os.makedirs('result/lwtw')
os.makedirs('dataresult/lwtw')
plt.savefig('result/lwtw/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='lw/tw', feature_name='lw/tw', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/lwtw/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='lw/tw', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'lw/tw', plot_pts_dist=True)

plt.savefig('result/lwtw/picture3.svg')


fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='lwtw', plot_pts_dist=True
)


X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)
X.to_excel(r'dataresult/lwtw/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/lwtw/Y.xlsx', sheet_name='train', index = False)


Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/lwtw/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/lwtw/picture4.svg')

# 10.Aspect ratio--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Aspect ratio', feature_name='Aspect ratio', target=titanic_target, show_percentile=True
)

os.makedirs('result/Aspect ratio')
os.makedirs('dataresult/Aspect ratio')
plt.savefig('result/Aspect ratio/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Aspect ratio', feature_name='Aspect ratio', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/Aspect ratio/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Aspect ratio', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Aspect ratio', plot_pts_dist=True)
plt.savefig('result/Aspect ratio/picture3.svg')



fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Aspect ratio', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/Aspect ratio/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Aspect ratio/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Aspect ratio/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/Aspect ratio/picture4.svg')

# 11.Af(mm^2)--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Af/Ag', feature_name='Af/Ag', target=titanic_target, show_percentile=True
)

os.makedirs('result/AfAg')
os.makedirs('dataresult/AfAg')
plt.savefig('result/AfAg/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Af/Ag', feature_name='Af/Ag', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/AfAg/picture2.svg')
# 2.3 pdp for feature 'embarked'

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Af/Ag', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'Af/Ag', plot_pts_dist=True)

plt.savefig('result/AfAg/picture3.svg')


fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='AfAg', plot_pts_dist=True
)


X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/AfAg/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/AfAg/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/AfAg/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/AfAg/picture4.svg')

# 12.P/fcAg--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='P/fcAg', feature_name='P/fcAg', target=titanic_target, show_percentile=True
)

os.makedirs('result/PfcAg')
os.makedirs('dataresult/PfcAg')
plt.savefig('result/PfcAg/picture1.svg')

# check prediction distribution through feature 'embarked'

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='P/fcAg', feature_name='P/fcAg', 
    show_percentile=True, predict_kwds={}
)

plt.savefig('result/PfcAg/picture2.svg')

pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='P/fcAg', num_grid_points=19
)

fig, axes = pdp.pdp_plot(pdp_fare, 'P/fcAg', plot_pts_dist=True)

plt.savefig('result/PfcAg/picture3.svg')


fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='P/fcAg', plot_pts_dist=True
)

X= pd.DataFrame(pdp_fare.display_columns)
Y= pd.DataFrame(pdp_fare.pdp)

X.to_excel(r'dataresult/PfcAg/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/PfcAg/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_fare.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/PfcAg/Z.xlsx', sheet_name='train', index = False)

plt.savefig('result/PfcAg/picture4.svg')

# 12.Section--------------------------------------------------------------------------------------------------------------------------------------------
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature=['Section_R', 'Section_B', 'Section_F'], feature_name='Section', 
    target=titanic_target
)
os.makedirs('result/Section')
os.makedirs('dataresult/Section')
plt.savefig('result/Section/picture1.svg')

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature=['Section_R', 'Section_B', 'Section_F'], 
    feature_name='Section', predict_kwds={}
)

plt.savefig('result/Section/picture2.svg')

titanic_data= titanic_data.astype(float)

pdp_embark = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, 
    feature=['Section_R', 'Section_B', 'Section_F']
)

fig, axes = pdp.pdp_plot(pdp_embark, 'Section')

X= pd.DataFrame(pdp_embark.display_columns)
Y= pd.DataFrame(pdp_embark.pdp)

X.to_excel(r'dataresult/Section/X.xlsx', sheet_name='train', index = False)
Y.to_excel(r'dataresult/Section/Y.xlsx', sheet_name='train', index = False)

Z= pd.DataFrame(pdp_embark.ice_lines)
Z=Z.T
Z.to_excel(r'dataresult/Section/Z.xlsx', sheet_name='train', index = False)
plt.savefig('result/Section/picture3.svg')
fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_fare, feature_name='Section', plot_pts_dist=True)
