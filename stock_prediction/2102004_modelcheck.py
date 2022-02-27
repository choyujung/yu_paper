import stock_function as sf
# from stock_function import *
import pandas as pd
import numpy as np
import matplotlib
from sklearn import metrics
import glob
import os
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests, coint
from statsmodels.tsa.api import VAR

import tensorflow
import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--phase', type=str, default='None', required=False)
parser.add_argument('--keyword', type=str, default='None', required=False)
# parser.add_argument('--keyword1', type=str, default='None', required=False)
# parser.add_argument('--keyword2', type=str, default='None', required=False)

args = parser.parse_args()


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
    df_merge = df_merge.loc[(df_merge['Date'] >= '2016-03-13') ] #& (df_merge['Date'] < '2017-10-15'), :]
# else:
#     df_merge = df_merge.loc[df_merge['Date'] >= '2017-10-15', :]

print('len:', len(df_merge))
# print(df_merge)
# p = ['ai', 'ml', 'dl', 'dm'] # ['ai', 'ml'], ['ai', 'dl'], ['ai', 'dm'], ['ml', 'dl', 'dm'], ['ai', 'ml', 'dl', 'dm']]

df = df_merge[[args.keyword, 'turnover', 'revised_index']]
df = df.dropna(axis=0, how='any')[[args.keyword, 'turnover', 'revised_index']]
df = df_merge[[args.keyword, 'turnover', 'revised_index']]
print(df.columns)


print(df.shape)

nobs = math.ceil(len(df) * 0.2)

df_array = df.values
x, y = sf.split_xy5(df_array, 4, 1)
x_train = x[:-nobs]
x_test = x[-nobs:]
y_train = y[:-nobs]
y_test = y[-nobs:]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# model.json 파일열기
json_file = open(path + '/final_result/'+ args.phase + args.keyword +"_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

from tensorflow.compat.v2.keras.models import model_from_json
# json파일로부 터 model 로드하기
loaded_model = model_from_json(loaded_model_json)

# 로드 한모델 에 weight 로드하기
loaded_model.load_weights(path + '/final_result/' + args.phase + args.keyword + '_model.h5')

# 모 델 컴파일
loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])

loss, mse = loaded_model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)
print('rmse : ', mse**0.5)

loaded_pred = loaded_model.predict(x_test)
predict_df = pd.DataFrame({'Date': df_merge['Date'][-len(y_test):],
                           'lstm_label': list(y_test), 'lstm_pred': list(loaded_pred)})

predict_df.to_csv(path + args.phase + args.keyword +'_loaded_pred.csv', encoding='utf-8-sig', index=False)


# 모델 성능 테스트
MSE = metrics.mean_squared_error(y_test, loaded_pred)
RMSE = metrics.mean_squared_error(y_test, loaded_pred)**0.5
print('loaded_RMSE:', RMSE)

mse_df = pd.DataFrame({'loss': loss, 'mse': mse, 'rmse': RMSE}, index=[0])
# mse_df.to_csv(path+args.phase + args.keyword + '_' +'_mse.csv', encoding='utf-8-sig', index=False)
mse_df.to_csv(path + args.phase + args.keyword +'_loaded_mse.csv', encoding='utf-8-sig', index=False)
print('====================================================')

plt.plot(range(len(y_test)), y_test, '.-', c='k' )
plt.plot(range(len(y_test)), loaded_pred, '--', c='b')
plt.ylabel('stock index')
plt.xlabel("time")
plt.legend(['Original', 'Predicted'])
# plt.savefig(path + '/' +args.phase + args.keyword + '_predicted.png')
plt.savefig(path + '/' + args.phase + args.keyword +'_loaded_predicted.png')