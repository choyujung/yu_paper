# https://medium.com/@kimkido95/time-series-analysis-using-var-6737cf2055cb
# https://predictor-ver1.tistory.com/3
# https://analysis-flood.tistory.com/40
# https://byeongkijeong.github.io/ARIMA-with-Python/
# https://post.naver.com/viewer/postView.nhn?volumeNo=28094462&memberNo=18071586
# https://www.statsmodels.org/dev/vector_ar.html
# https://mybeta.tistory.com/27
# https://blog.naver.com/chunjein/221589590358
# https://medium.com/@kimkido95/time-series-analysis-using-var-6737cf2055cb

import pandas as pd
import numpy as np
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests

# Data Scaling
def scaleColumns(df, cols_to_scale):
    # dataframe에 있는 데이터 정규화 처리
    scaler = preprocessing.MinMaxScaler()
    for col in cols_to_scale:
        df[col] = scaler.fit_transform(pd.DataFrame(df[col]))
    return df

# Granger causality test matrix
def grangers_causality_matrix(X_train, variables, test='ssr_chi2test', verbose=False):
    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(X_train[[r, c]], maxlag= maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            # if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')     #...?
            min_p_value = np.min(p_values)
            dataset.loc[r, c] = min_p_value
    dataset.columns = [var + '_x' for var in variables]
    dataset.index = [var + '_y' for var in variables]
    return dataset

# ADF Test
def adf_test(series, signif=0.05):
    result = adfuller(series, autolag='AIC')
    adf = pd.Series(result[0:4], index=['Test Statistic', 'p-value', 'Lags', 'Observations'])
    for key, value in result[4].items():
        adf['Critical Value (%s)' % key] = value
    print(adf)

    p = adf['p-value']
    if p <= signif:
        print("Series is Stationary")
    else:
        print("Series is Non-Stationary")

def CalculateEWMAVol (ReturnSeries, Lambda):
    SampleSize = len(ReturnSeries)
    Average = ReturnSeries.mean()

    e = np.arange(SampleSize-1,-1,-1)        # np.arange(start,stop,step)
    r = np.repeat(Lambda,SampleSize)
    vecLambda = np.power(r,e)

    sxxewm = (np.power(ReturnSeries-Average,2)*vecLambda).sum()
    Vart = sxxewm/vecLambda.sum()
    EWMAVol = math.sqrt(Vart)

    return (EWMAVol)


# Invert the transformation to get the real forecast
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


# LSTM 데이터셋 5일씩 분리
def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number,1]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(label.iloc[i + window_size]))
    return np.array(feature_list), np.array(label_list)
