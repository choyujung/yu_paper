
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
parser.add_argument('--thema', type=str, default='None', required=False)
#parser.add_argument('--keyword_file', type=str, default='None', required=False)
#parser.add_argument('--lag', type=int, default='None', required=False)
parser.add_argument('--phase', type=str, default='None', required=False)
parser.add_argument('--target', type=str, default='None', required=False)

# parser.add_argument('--batch_size1', type=int, required=True)
# parser.add_argument('--batch_size2', type=int, required=True )

args = parser.parse_args()

# os.chdir("/home/caitech/Desktop/yujung/stock_research")
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/


# dataframe 에 있는 데이터 정규화 처리
scaler = preprocessing.MinMaxScaler()

# 폴더에 있는 모든 주식데이터 파일를 불어오기
path = '/home/caitech/Desktop/yujung/stock_research/datasets/주가데이터_주별/' + args.thema +'/'+args.phase
path_key = '/home/caitech/Desktop/yujung/stock_research/datasets/키워드_주별/'
file = glob.glob(os.path.join(path, "*.csv"))
file_key = glob.glob(os.path.join(path_key, "*.csv"))


# # 4차산업, naver 단일키워드 검색량 데이터
# for k in range(len(file_key)):
#
#     keyword = pd.read_csv(file_key[k], engine='python')
#     keyword = keyword.reset_index()[1:]
#     keyword.columns = ['Date', 'count']
#     keyword.reset_index(drop=True, inplace=True)
#     keyword['Date'] = pd.to_datetime(keyword['Date'], format="%Y-%m-%d")
#     keyword_scale = sf.scaleColumns(keyword, ['count'])
#     keyword_scale.dropna(axis=0, how='any')
#
#     keyword_name = file_key[k].split('/')[-1].split('_')[0]
#
#     print("[", keyword_name, "]")
#
#     # d = timedelta(days=2)

key = []
company = []
var_rmse = []
lstm_rmse = []
# lag =[]

for i in range(len(file)):
    # keyword data
    for k in range(len(file_key)):
        keyword = pd.read_csv(file_key[k], engine='python')
        keyword = keyword.reset_index()[1:]
        keyword.columns = ['Date', 'count']
        keyword.reset_index(drop=True, inplace=True)
        keyword['Date'] = pd.to_datetime(keyword['Date'], format="%Y-%m-%d")
        keyword_scale = sf.scaleColumns(keyword, ['count'])
        keyword_scale.dropna(axis=0, how='any')

        keyword_name = file_key[k].split('/')[-1].split('_')[0]

        print("[", keyword_name, "]")
        key.append(keyword_name)

        # stock data
        f = pd.read_csv(file[i])
        f = f.iloc[:, 1:]

        f = f.loc[f['High'] != 0]

        # 날짜 형식 바꾸기
        d = timedelta(days=2)

        f = f.astype({'Date': 'str'})

        f['Date'] = pd.to_datetime(f['Date'], format="%Y-%m-%d")
        f['Date'] = f['Date']+d

        # 주식데이터 scaling
        # f_scale = sf.scaleColumns(f, [args.target])
        # f_scale = f.dropna(axis=0, how='any')[['Date', args.target]]

        f_scale = sf.scaleColumns(f, [['Close', 'Volume', 'High', 'Low']])
        f_scale = f.dropna(axis=0, how='any')[['Date', 'Close', 'Volume', 'High', 'Low']]



        df_merge = pd.merge(f_scale, keyword_scale, on="Date", how="left")
        df_merge = df_merge.loc[df_merge['Date'] < '2016-03-08', :]
        # df_merge = df_merge.loc[(df_merge['Date'] > '2016-03-08')&(df_merge['Date'] < '2018-01-31'), :]
        # df_merge = df_merge.loc[df_merge['Date'] > '2018-01-31', :]
        df = df_merge[['Volume', 'High', 'Low','count', args.target]]


        # 회사명 출력
        company_name = file[i].split('/')[-1].split('.')[0]
        company.append(company_name)
        print("[", company_name, "]")

        nobs = math.ceil(len(df) * 0.3)              # 다음 몇개의 관측치를 예측하는데 사용할 것인지

        train = df[:math.ceil(-nobs)]
        test = df[math.ceil(-nobs):]

        # LSTM
        # 데이터 처리
        #feature_cols = ['Open', 'High', 'Low', 'Volume']
        # feature_cols = [args.keyword_file.split('.')[0]]
        # feature_cols = ['count']
        # label_cols = [args.target]
        #
        # # train_feature
        # train_feature = train[feature_cols]
        # # train_label
        # train_label = train[label_cols]
        #
        # test_feature = test[feature_cols]
        # test_label = test[label_cols]
        #
        # train_feature, train_label = sf.make_dataset(train_feature, train_label, args.lag)
        # test_feature, test_label = sf.make_dataset(test_feature, test_label, args.lag)


        df_array = df.values
        x, y = sf.split_xy5(df_array, 4, 1)
        x_train = x[:-nobs]
        x_test = x[-nobs:]
        y_train = y[:-nobs]
        y_test = y[-nobs:]

        # x_train = train_feature[:-nobs]
        # x_test = train_feature[-nobs:]
        #
        # y_train = train_label[:-nobs]
        # y_test = train_label[-nobs:]

        # print(train_feature.shape, train_label.shape)
        # print(test_feature.shape, test_label.shape)

        print(x_train .shape, x_test.shape)
        print(y_train.shape, y_test.shape)

        # # 모델구성
        # model = Sequential()
        # model.add(LSTM(64, input_shape=(train_feature.shape[1], train_feature.shape[2])))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(1))
        # model.summary()
        #
        # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        #
        # early_stopping = EarlyStopping(patience=20)
        # model.fit(train_feature, train_label, validation_split=0.2, verbose=1,
        #           batch_size=1, epochs=100, callbacks=[early_stopping])
        #
        # loss, mse = model.evaluate(test_feature, test_label, batch_size=1)  ##??
        # print('loss : ', loss)
        # print('mse : ', mse)
        #
        # y_pred = model.predict(test_feature)

        # 모델구성
        model = Sequential()
        model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit(x_train, y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])

        loss, mse = model.evaluate(x_test, y_test, batch_size=1)
        print('loss : ', loss)
        print('mse : ', mse)

        globals()['y_pred'+str(k)] = model.predict(x_test)


        predict_df = pd.DataFrame({'Date': df_merge['Date'][-len(y_test):],
                                   'lstm_label': list(y_test), 'lstm_pred': list(globals()['y_pred'+str(k)])})
        predict_df.to_excel(path+keyword_name+company_name+'_pred.xlsx')


        # 모델 성능 테스트
        MSE = metrics.mean_squared_error(y_test, globals()['y_pred'+str(k)])
        RMSE = metrics.mean_squared_error(y_test, globals()['y_pred'+str(k)])**0.5
        lstm_rmse.append(RMSE)
        print('LSTM_RMSE:', RMSE)
        print('====================================================')

    plt.figure()
    # plt.plot(range(len(y_test)), y_test, '.-', c='#1f77b4', )
    # plt.plot(range(len(y_test)), y_pred0, '--', c='#ff7f0e')
    # plt.plot(range(len(y_test)), y_pred1, '-.', c='#bcbd22')
    # plt.plot(range(len(y_test)), y_pred2, '-', c='#e377c2')
    # plt.plot(range(len(y_test)), y_pred3, ':', c='#8c564b')
    plt.plot(range(len(y_test)), y_test, '.-', c='k' )
    plt.plot(range(len(y_test)), y_pred0, '-', c='b')
    plt.plot(range(len(y_test)), y_pred1, '--', c='g')
    plt.plot(range(len(y_test)), y_pred2, '-.', c='m')
    plt.plot(range(len(y_test)), y_pred3, ':', c='r')
    plt.xticks(range(len(y_test)))
    plt.ylabel("Close")
    plt.xlabel("time")
    plt.legend([args.target, 'AI', 'ML', 'DL', 'DM'])
    plt.savefig(path + '/' + args.phase + company_name + args.target + '.png')

# 결과 저장

final = {"회사명": company,
         "key_word": key,
         "LSTM_RMSE": lstm_rmse,
         }
final_df = pd.DataFrame(final)
final_df.to_csv('/home/caitech/Desktop/yujung/stock_research/'+args.phase+'_'+args.target+'.csv', encoding='utf-8-sig', index=False)


# plt.figure()
# plt.plot(range(len(y_test)), y_test, 'o-', c='black',)
# plt.plot(range(len(y_test)), y_pred0, '^--', c='gray')
# plt.plot(range(len(y_test)), y_pred1, 's--', c='gray')
# plt.plot(range(len(y_test)), y_pred2, 'D--', c='gray')
# plt.plot(range(len(y_test)), y_pred3, 'p--', c='gray')
# plt.xticks(range(len(y_test)))
# plt.ylabel("Close")
# plt.xlabel("time")
# plt.legend([args.target, 'AI', 'ML', 'DL', 'DM'])
# plt.savefig(path+'/'+ args.phase + company_name + args.target +'.png')




# 결과 저장
#
# final = {"회사명": company,
#          "key_word" : key,
#         "LSTM_RMSE": lstm_rmse,
#
#          }
# final_df = pd.DataFrame(final)
# final_df.to_csv('/home/caitech/Desktop/yujung/stock_research/'+args.phase+keyword_name+'_'+args.target+'.csv', encoding='utf-8-sig', index=False)







