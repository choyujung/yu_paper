import os
import stock_function as sf
import pandas as pd
import numpy as np
import math
from sklearn import metrics
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima_model import ARIMA
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--phase', type=str, default='None', required=False)
parser.add_argument('--keyword', type=str, default='None', required=False)

args = parser.parse_args()


# 폴더에 있는 모든 주식데이터 파일를 불어오기
path = '/home/caitech/Desktop/yujung/stock_research/datasets/주가데이터_주별/'
# path = 'G:/내 드라이브/Ubuntu/yujung/stock_research/'
path_key = '/home/caitech/Desktop/yujung/stock_research/datasets/키워드_주별/'
# path_key = 'G:/내 드라이브/Ubuntu/yujung/stock_research/'
# file = glob.glob(os.path.join(path, "*.xlsx"))
# file_key = glob.glob(os.path.join(path_key, "*.csv"))

keyword = pd.read_excel(path+'googletrends_주별_통합.xlsx')
keyword['Date'] = pd.to_datetime(keyword['Date'], format="%Y-%m-%d")
keyword_scale = sf.scaleColumns(keyword, [['ai', 'ml', 'dl', 'dm']])
keyword_scale.dropna(axis=0, how='any')

f = pd.read_excel(path+'주가지수_회전율.xlsx')
d = timedelta(days=2)

f = f.astype({'Date': 'str'})

f['Date'] = pd.to_datetime(f['Date'], format="%Y-%m-%d")
f['Date'] = f['Date']+d

# scaling
f_scale = sf.scaleColumns(f, [['revised_index', 'turnover']])

f_scale = f.dropna(axis=0, how='any')[['Date', 'revised_index', 'turnover']]

df_merge = pd.merge(f_scale, keyword_scale, on="Date", how="left")

if args.phase == 'phase1':
    df_merge = df_merge.loc[df_merge['Date'] < '2016-03-13', :]
elif args.phase == 'phase2':
    df_merge = df_merge.loc[(df_merge['Date'] >= '2016-03-13')]# & (df_merge['Date'] < '2017-10-15'), :]
# else:
#     df_merge = df_merge.loc[df_merge['Date'] >= '2017-10-15', :]

print('len:', len(df_merge))
# print(df_merge)
# p = ['ai', 'ml', 'dl', 'dm'] # ['ai', 'ml'], ['ai', 'dl'], ['ai', 'dm'], ['ml', 'dl', 'dm'], ['ai', 'ml', 'dl', 'dm']]


df = df_merge[['ai', 'ml', 'dl', 'turnover', 'revised_index']]
df = df.dropna(axis=0, how='any')[['ai', 'ml', 'dl', 'turnover', 'revised_index']]
df = df_merge[['ai', 'ml', 'dl', 'turnover', 'revised_index']]
print(df.columns)
# nobs = math.ceil(len(df) * 0.2)
#
# df_array = df.values
# x, y = sf.split_xy5(df_array, 4, 1)
# x_train = x[:-nobs]
# x_test = x[-nobs:]
# y_train = y[:-nobs]
# y_test = y[-nobs:]
#
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

nobs = math.ceil(len(df) * 0.2)  # 다음 몇개의 관측치를 예측하는데 사용할 것인지

train = df[:math.ceil(-nobs)]
test = df[math.ceil(-nobs):]
print("train_len : ", len(train))
print("test_len : ", len(test))

# ADF Test
result = adfuller(df['ai'], autolag='AIC')
adf_test = result[0]
adf_pvalue = result[1]
# print("ADF test statistic: {}".format(result[0]))
# print("p-value: {}".format(result[1]))


df_differenced = df.diff(1).dropna()
train_differenced = df_differenced[:math.ceil(-nobs)]
test_differenced = df_differenced[math.ceil(-nobs):]


# # 1st difference 데이터프레임 모든 열 차분
# train_differenced = train.diff(1).dropna()
# test_differenced = test.diff(1).dropna()
# print(train_differenced)

# ADF Test Again
result2 = adfuller(df_differenced['ai'], autolag='AIC')
adf_test2 = result2[0]
adf_pvalue2 = result2[1]
print("ADF test statistic: {}".format(result2[0]))
print("p-value: {}".format(result2[1]))
# sf.adf_test(train_differenced['index'])
# # sf.adf_test(train_differenced['daily_vol'])

print("------------------------------")

# VAR model
model = VAR(train_differenced)

# AIC
# aic_list = []
# for i in range(1, 50):
#     result = model.fit(i)
#     aic_list.append(result.aic)
#
# min_aic = aic_list.index(min(aic_list))#+1        ######

# VAR model fitting                                          #####
VAR_result = model.fit(maxlags=4)             #### maxlags=min_aic,

# Get the lag order
# lag_order = VAR_result.k_ar
# if lag_order == 0:
#     lag_order = 1
# else:
#     lag_order = VAR_result.k_ar
lag_order = VAR_result.k_ar

# lag.append(lag_order)
print('lag_order : ', lag_order)

# VAR forecast
forecast_input = train_differenced.values[-lag_order:]

fc = VAR_result.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns = df.columns + '_1d')
# fc_df = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_1d')
# print(fc_df)
#fc_df = pd.DataFrame(fc, columns=df.columns + '_forecast')

# VAR result
df_results = sf.invert_transformation(train, df_forecast)
print(df_forecast)
# df_forecast = df_forecast.loc[:, ['ai', 'turnover', 'index']]

# fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
# for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
#     df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
#     test[col][-nobs:].plot(legend=True, ax=ax);
#     ax.set_title(col + ": Forecast vs Actuals")
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.spines["top"].set_alpha(0)
#     ax.tick_params(labelsize=6)

# plt.tight_layout()
# plt.show()

# print(VAR_result.summary())
# VAR_result.plot_forecast(nobs)
# plt.tight_layout()
# plt.show()

y = test.values[:, -1]
y_hat = df_results.values[:, -1]
#
# y = test.values[:, 1]
# y_hat = fc_df.values[:, 1]
#
#
# var_pred = pd.DataFrame({'Date': df_merge['Date'][-len(y):],
#                            'var_label': y, 'var_pred': y_hat})
# var_pred.to_excel(path+company_name+'var_pred.xlsx')
#
#
MSE = metrics.mean_squared_error(y, y_hat)
RMSE = metrics.mean_squared_error(y, y_hat) ** 0.5
#
# var_rmse.append(RMSE)
print('VAR_RMSE:', RMSE)
# print('====================================================')

mse_df = pd.DataFrame({'adf_pvalue': adf_pvalue, 'adf_pvalue2': adf_pvalue2, 'mse': MSE, 'rmse': RMSE}, index=[0])
mse_df.to_csv(path + '/' + args.phase + args.keyword+'_VARmse.csv', encoding='utf-8-sig', index=False)

# ARIMA model

