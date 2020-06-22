import xgboost as xgb
import pandas as pd
import seaborn as sns
import numpy as np
from geopy.distance import geodesic 
from feature_selector import FeatureSelector
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Вычисляет азимут по двум заданным точкам, код взят отсюда: https://pastebin.com/PHeWmiEN
def get_azimuth(rad,city_lat,city_long, latitude, longitude):
    llat1 = city_lat
    llong1 =city_long
    llat2 = latitude
    llong2 = longitude

    lat1 = llat1*math.pi/180.
    lat2 = llat2*math.pi/180.
    long1 = llong1*math.pi/180.
    long2 = llong2*math.pi/180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
    x = sl1*sl2+cl1*cl2*cdelta
    ad = math.atan2(y,x)

    x = (cl1*sl2) - (sl1*cl2*cdelta)
    y = sdelta*cl2
    z = math.degrees(math.atan(-y/x))

    if (x < 0):
        z = z+180.

    z2 = (z+180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
    angledeg = (anglerad2*180.)/math.pi
    
    return round(angledeg, 2)

#Вычисляет среднюю абсолютную процентную ошибку
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Вычисляет медианную абсолютную процентную ошибку
def median_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

#Печатает рассчитанные значения коэффициента детерминации, средней и медианной абсолютных ошибок
def print_metrics(prediction, val_y):
    val_mae = mean_absolute_error(val_y, prediction)
    median_AE = median_absolute_error(val_y, prediction)
    rmse = math.sqrt(mean_squared_error(val_y, prediction))
    r2 = r2_score(val_y, prediction)

    print('')
    print('R\u00b2: {:.2}'.format(r2))
    print('RMSE: {:.2}'.format(rmse))
    print('Средняя абсолютная ошибка: {:.3} %'.format(mean_absolute_percentage_error(val_y, prediction)))
    print('Медианная абсолютная ошибка: {:.3} %'.format(median_absolute_percentage_error(val_y, prediction)))

#При помощи библиотеки pandas считываем csv-файл и преобразуем его в формат датафрейма (таблицы)
file_path = 'C:/Users/ilyaa/Desktop/dat2.csv'
df_I = pd.read_csv(file_path)

#Выводим 5 первых строк датафрейма
print(df_I.head(5))

#Создаем новый столбец Стоимость 1 кв.м путем построчного деления стоимостей квартир на их общие площади
df_I['priceMetr'] = df_I['price']/df_I['totalArea']

#Задаем широту и долготу центра города и рассчитываем для каждой квартиры расстояние от центра и азимут 
Ivanovo_center_coordinates = [57.000348, 40.973921]
df_I['distance'] = list(map(lambda x, y: geodesic(Ivanovo_center_coordinates, [x, y]).meters, df_I['latitude'], df_I['longitude']))
df_I['azimuth'] = list(map(lambda x, y: get_azimuth(6372795,Ivanovo_center_coordinates[0],Ivanovo_center_coordinates[1],x, y), df_I['latitude'], df_I['longitude']))

#Выбираем из датафрейма только те квартиры, которые расположены не дальше 40 км от центра города с панельными стенами
df_I = df_I.loc[(df_I['distance'] < 40000)] 

#Округляем значения стоблцов Стоимости метра, расстояния и азимута
df_I['priceMetr'] = df_I['priceMetr'].round(0)
df_I['distance'] = df_I['distance'].round(0)
df_I['azimuth'] = df_I['azimuth'].round(0)

#Выводим сводную информацию о датафрейме и его столбцах (признаках)
df_I.info()

mediane = df_I['priceMetr'].quantile(q=0.5)
print(f'Медиана: {mediane}')
#Вычисляем строки со значениями-выбросами 
first_quartile = df_I['priceMetr'].quantile(q=0.4)
print(f'Нижняя граница: {first_quartile}')
third_quartile = df_I['priceMetr'].quantile(q=0.6)
print(f'Верхняя граница: {third_quartile}')
outliers = df_I[(df_I['priceMetr'] > third_quartile) | (df_I['priceMetr'] < first_quartile )].count(axis=1)

 
# Удаляем из датафрейма 3000 строк, подходящих под критерии выбросов
print(f'Кол-во выбросов: {outliers.count}')
df_I.drop(outliers.index, inplace=True)

df_I.drop(['latitude'], axis=1,inplace = True)
df_I.drop(['longitude'], axis=1,inplace = True)
df_I.drop(['price'], axis=1,inplace = True)
#Выводим сводную информацию о датафрейме и его столбцах (признаках)
df_I.info()
cor = df_I.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Назначаем целевой переменной цену 1 кв. метра, а можно и цену всей квартиры, тогда будет y = df['price']
y_I = df_I['priceMetr']

#Создаем список признаков, на основании которых будем строить модели
features = [
            'floorNumber', 
            'floorsTotal', 
            'totalArea', 
            'distance',
            'azimuth',
           ]

#Создаем датафрейм, состоящий из признаков, выбранных ранее
X_I = df_I[features]

#Проводим случайное разбиение данных на выборки для обучения (train) и валидации (val), по умолчанию в пропорции 0.75/0.25
train_X, val_X, train_y, val_y = train_test_split(X_I, y_I, random_state=1)
 
 #Создаем регрессионную модель случайного леса 


# Создаем модели
rf_model = RandomForestRegressor(n_estimators=2000, 
                                 n_jobs=-1,  
                                 bootstrap=False,
                                 criterion='mae',
                                 max_features=2,
                                 random_state=1,
                                 max_depth=50,
                                 min_samples_split=5
                                 )
xgb_model = xgb.XGBRegressor(objective ='reg:gamma', 
                             learning_rate = 0.01,
                             max_depth = 50, 
                             n_estimators = 2000,
                             nthread = -1,
                             eval_metric = 'gamma-nloglik', 
                             )
reg_model = LinearRegression()

#Проводим подгонку модели на обучающей выборке 
rf_model.fit(train_X, train_y)
xgb_model.fit(train_X, train_y)
reg_model.fit(train_X, train_y)

#Вычисляем предсказанные значения цен на основе валидационной выборки
rf_prediction = rf_model.predict(val_X).round(0)
xgb_prediction = xgb_model.predict(val_X).round(0)
reg_prediction = reg_model.predict(val_X).round(0)

#Вычисляем и печатаем величины ошибок при сравнении известных цен квартир из валидационной выборки с предсказанными моделью
print("Величины ошибок для случайного леса")
print_metrics(rf_prediction, val_y)
print()
print("Величины ошибок для бустинга")
print_metrics(xgb_prediction, val_y)
print()
print("Величины ошибок для линейной регрессии")
print_metrics(reg_prediction, val_y)
print()

# #Усредняем предсказания обоих моделей
# prediction = rf_prediction * 0.5 + xgb_prediction * 0.5 

# #Вычисляем и печатаем величины ошибок для усредненного предсказания
# print("Величины ошибок для усредненного предсказания")
# print_metrics(prediction, val_y)

#Рассчитываем важность признаков в модели Random forest
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

#Печатаем рейтинг признаков
print()
print("Рейтинг важности признаков:")
for f in range(X_I.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
 
df_0 = pd.DataFrame({'Actual': val_y, 'Predicted': rf_prediction})
df1 = df_0
df1.plot(kind='bar', figsize=(10, 8))
plt.title('Случайный лес')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

df_0 = pd.DataFrame({'Actual': val_y, 'Predicted': xgb_prediction})
df1 = df_0
df1.plot(kind='bar', figsize=(10, 8))
plt.title('Бустинг')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

df_0 = pd.DataFrame({'Actual': val_y, 'Predicted': reg_prediction})
df1 = df_0
df1.plot(kind='bar', figsize=(10, 8))
plt.title('Линейная регрессия')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Строим столбчатую диаграмму важности признаков
plt.figure()
plt.title("Важность признаков")
plt.bar(range(X_I.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
plt.xticks(range(X_I.shape[1]), indices)
plt.xlim([-1, X_I.shape[1]])
plt.show()

#Создаем датафрейм с параметрами квартиры https://ivanovo.cian.ru/sale/flat/234203151/,
flat = pd.DataFrame({
                     'floorNumber':[3],
                     'floorsTotal':[5],
                     'totalArea':[47.6],
                     'latitude':[56.9704019],
                     'longitude':[40.9945961],
                  })


# #Вычисляем предсказанное значение стоимости по двум моделям
rf_prediction_flat = rf_model.predict(flat).round(0)
xgb_prediction_flat = xgb_model.predict(flat).round(0)
reg_prediction_flat = reg_model.predict(flat).round(0)
rf_price_flat = rf_prediction_flat*flat['totalArea'][0]
xgb_price_flat = xgb_prediction_flat*flat['totalArea'][0]
reg_price_flat = reg_prediction_flat*flat['totalArea'][0]
mean_price_flat = (rf_prediction_flat * 0.3 + xgb_prediction_flat * 0.3+ reg_prediction_flat * 0.3)*flat['totalArea'][0]
print(f'Предсказанная случайным лесом цена предложения: {int(rf_price_flat.round(-3))} рублей')
print(f'Предсказанная бустингом цена предложения: {int(xgb_price_flat.round(-3))} рублей')
print(f'Предсказанная регерссией цена предложения: {int(reg_price_flat.round(-3))} рублей')
print(f'Усредненная цена предложения: {int(mean_price_flat.round(-3))} рублей')