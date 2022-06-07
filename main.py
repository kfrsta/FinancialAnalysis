import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np  # библиотека предназначенная для расширения операций над объектами
import pandas as pd  # работа с dataframe
import matplotlib.pyplot as plt  # визуализация
from pandas.plotting import scatter_matrix  # дополнение визуализации
import yfinance as yf  # получение финансовых данных из интернета
import statsmodels.api as sm  # составление линейной регрессии
from prophet import Prophet


def viborka():
    print("___________________________________________________")
    print("_________Выборка 2021-05-07__2022-05-07____________")
    data2 = yf.download('NVTK.ME', '2021-05-07')  # скачиваем данные по выборке за год
    print(data2['Adj Close'].describe())  # вывод среднего аримфет, мин, макс знач, отклонения, медианы
    print("мода= ", data2['Adj Close'].mode(), sep=" ")  # высчитываем моду
    print("дисперсия= ", data2['Adj Close'].var(), sep=" ")  # высчитываем дисперсию
    dohodnost(data2[['Adj Close']])  # расчитываем доходность для выборки
    rolling(data2)  # строим скользящие средние для выборки
    print("____________________________________________________")


def regression_b_model():
    NVTK = yf.download('NVTK.ME')  # скачиваем котировки новатэк
    MOEX = yf.download('IMOEX.ME')  # скачиваем котировки мосбиржи
    NVTK = NVTK.resample('BM').apply(lambda x: x[-1])  # преобразуем для месячного периода
    MOEX = MOEX.resample('BM').apply(lambda x: x[-1])
    monthly_prices = pd.concat([NVTK['Close'], MOEX['Close']], axis=1)  # объединяем
    monthly_prices.columns = ['NVTK', 'IMOEX.ME']  # задаем индексы
    monthly_returns = monthly_prices.pct_change(1)
    clean_monthly_returns = monthly_returns.dropna(axis=0)
    X = clean_monthly_returns['IMOEX.ME']  # задаем параметры модели
    y = clean_monthly_returns['NVTK']  # у-зависимая переменная х независимая
    X1 = sm.add_constant(X)
    model_NVTK = sm.OLS(y, X1)  # строим модель
    results_NVTK = model_NVTK.fit()
    print(results_NVTK.summary())  # выводим отчёт о показателях модели


def dohodnost(daily_close):
    daily_pct_change = daily_close / daily_close.shift(1) - 1  # расчитываем доходность
    daily_pct_change.hist(bins=50)  # строим гистограмму доходности
    plt.show()
    print(daily_pct_change.describe())  # вывод среднего аримфет, мин, макс знач, отклонения, медианы


def predict_price(df):
    a=df['Adj Close']
    b=df.index
    a.reset_index()
    print(a.head())


def rolling(NVTK):
    NVTK['10'] = NVTK['Adj Close'].rolling(window=10).mean()  # строим скользящую среднюю сглаживание 40
    NVTK['200'] = NVTK['Adj Close'].rolling(window=200).mean()  # тоже самое сглаживание 100
    NVTK[['Adj Close', '10', '200']].plot(figsize=(20, 20))  # общий график
    plt.show()


print("___________________________________________________")
data = yf.download('NVTK.ME', '2004-01-01', '2022-05-17')  # скачиваем данные
# print(data.head())
#predict_price(data)
data['Adj Close'].plot()#показываем график цен акций
print(data['Adj Close'].describe()) #выводим статистические показатели
print("мода= ", data['Adj Close'].mode(), sep=" ") #считаем моду
print("дисперсия= ", data['Adj Close'].var(), sep=" ")# и дисперсию
dohodnost(data[['Adj Close']])
rolling(data)
regression_b_model()# вызов вышеописанных функций
viborka()
