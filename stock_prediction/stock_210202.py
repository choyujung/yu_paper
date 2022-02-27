
# from stock_function import *
# from typing import List

import stock_function as sf
# from stock_function import *
import pandas as pd
import numpy as np
import matplotlib
from sklearn import metrics
import glob
import os
import math
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests, coint
from statsmodels.tsa.api import VAR

import tensorflow
import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import argparse
parser = argparse.ArgumentParser()
# # parser.add_argument('--thema', type=str, default='None', required=False)
# #parser.add_argument('--keyword_file', type=str, default='None', required=False)
# #parser.add_argument('--lag', type=int, default='None', required=False)
parser.add_argument('--phase', type=str, default='None', required=False)
parser.add_argument('--keyword1', type=str, default='None', required=False)
parser.add_argument('--keyword2', type=str, default='None', required=False)
parser.add_argument('--batch_size', type=int, default='None', required=False)
# parser.add_argument('--target', type=str, default='None', required=False)
#
# # parser.add_argument('--batch_size1', type=int, required=True)
# # parser.add_argument('--batch_size2', type=int, required=True )
#
args = parser.parse_args()

# os.chdir("/home/caitech/Desktop/yujung/stock_research")
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/


# dataframe 에 있는 데이터 정규화 처리
# scaler = preprocessing.MinMaxScaler()

# 폴더에 있는 모든 주식데이터 파일를 불어오기
path = '/home/caitech/Desktop/yujung/stock_research/datasets/주가데이터_주별/'
path_key = '/home/caitech/Desktop/yujung/stock_research/datasets/키워드_주별/'
# file = glob.glob(os.path.join(path, "*.xlsx"))
# file_key = glob.glob(os.path.join(path_key, "*.csv"))

keyword = pd.read_excel(path+'googletrends_주별_통합.xlsx')
keyword['Date'] = pd.to_datetime(keyword['Date'], format="%Y-%m-%d")
keyword_scale = sf.scaleColumns(keyword, [['ai', 'ml', 'dl', 'dm']])
keyword_scale.dropna(axis=0, how='any')

#keyword_name= keyword.columns[]

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

# df = df_merge[[args.keyword, 'turnover', 'index']]
# df = df.dropna(axis=0, how='any')[[args.keyword, 'turnover','index']]
# df = df_merge[[args.keyword, 'turnover', 'index']]

df = df_merge[['ai', args.keyword1, args.keyword2, 'turnover', 'revised_index']]
df = df.dropna(axis=0, how='any')[['ai', args.keyword1, args.keyword2, 'turnover', 'revised_index']]
df = df_merge[['ai', args.keyword1, args.keyword2, 'turnover', 'revised_index']]
print(df.shape)
print(df.columns)

nobs = math.ceil(len(df) * 0.2)

df_array = df.values
x, y = sf.split_xy5(df_array, 4, 1)
x_train = x[:-nobs]
x_test = x[-nobs:]
y_train = y[:-nobs]
y_test = y[-nobs:]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

model = Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

early_stopping = EarlyStopping(patience=20)

start_time = time.time()

model.fit(x_train, y_train, validation_split=0.2, verbose=1,
          batch_size=args.batch_size, epochs=100, callbacks=[early_stopping])

run_time = time.time()-start_time
print("RUN_TIME :", run_time)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

# save model with json format
model_json = model.to_json()
with open(path + '/' + args.phase + 'all_' + 'b' + str(args.batch_size) +"_model.json", "w") as json_file:
    json_file.write(model_json)

# Save weight with h5 format
model.save_weights(path + '/'+args.phase + 'all_' + 'b' + str(args.batch_size) +'_model.h5')


y_pred = model.predict(x_test)


predict_df = pd.DataFrame({'Date': df_merge['Date'][-len(y_test):],
                           'lstm_label': list(y_test), 'lstm_pred': list(y_pred)})

# predict_df.to_csv(path + args.phase + args.keyword + '_' +'_pred.csv', encoding='utf-8-sig', index=False)
predict_df.to_csv(path + args.phase + 'all_' + 'b' + str(args.batch_size) +'_pred.csv', encoding='utf-8-sig', index=False)
# args.keyword1 + '_' + args.keyword2

# 모델 성능 테스트
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = metrics.mean_squared_error(y_test, y_pred)**0.5
print('LSTM_RMSE:', RMSE)

mse_df = pd.DataFrame({'loss': loss, 'mse': mse, 'rmse': RMSE, 'run_time': run_time}, index=[0])
# mse_df.to_csv(path+args.phase + args.keyword + '_' +'_mse.csv', encoding='utf-8-sig', index=False)
mse_df.to_csv(path + args.phase + 'all_'+ 'b' + str(args.batch_size) +'_mse.csv', encoding='utf-8-sig', index=False)
print('====================================================')

plt.plot(range(len(y_test)), y_test, '.-', c='k' )
plt.plot(range(len(y_test)), y_pred, '--', c='b')
plt.ylabel('stock index')
plt.xlabel("time")
plt.legend(['Original', 'Predicted'])
# plt.savefig(path + '/' +args.phase + args.keyword + '_predicted.png')
plt.savefig(path + '/' + args.phase + 'all_'+'b' + str(args.batch_size) + '_predicted.png')

