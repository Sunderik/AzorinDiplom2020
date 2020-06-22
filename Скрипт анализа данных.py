import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from subprocess import check_output

data1 = pd.read_csv('исходные данные.csv',encoding = "cp1251")
categorical_columns = [c for c in data1.columns if data1[c].dtype.name == 'object']
numerical_columns   = [c for c in data1.columns if data1[c].dtype.name != 'object']
print("Категориальные признаки {}".format(categorical_columns))
print("Количественные признаки {}".format(numerical_columns))
print()
print()
data = pd.read_csv('исходные данные.csv',encoding = "cp1251")
data_labels = data.iloc[ :, [ 0, 32]]
print(data)
print()
print(data.corr())

print()
from feature_selector import FeatureSelector
 
# Признаки - в train, метки - в train_labels
fs = FeatureSelector(data = data, labels = data_labels)
fs.identify_missing(missing_threshold=0.6)
print(fs.missing_stats.head())
	
plt.show(fs.plot_missing())

fs.identify_collinear(correlation_threshold=0.01)
correlated_features = fs.ops['collinear']
correlated_features[:29]
plt.show(fs.plot_collinear())

# список признаков для удаления
collinear_features = fs.ops['collinear']
 
# датафрейм коллинеарных признаков
print(fs.record_collinear.head())
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)
one_hot_features = fs.one_hot_features
base_features = fs.base_features
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))
print(fs.data_all.head(10))
zero_importance_features = fs.ops['zero_importance']
plt.show(fs.plot_feature_importances(threshold = 0.90, plot_n = 7))
fs.identify_single_unique()
plt.show(fs.plot_unique())
