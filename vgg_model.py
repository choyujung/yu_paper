import os
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import shutil
import argparse
import math
import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator ,array_to_img, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models,layers, Input, optimizers, initializers, regularizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,  EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import classification_report, confusion_matrix
# from glob import glob
# import PIL
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# from os import listdir
# from os.path import isfile, join
# print(tf.__version__)
# print(keras.__version__)
os.chdir('/home/caitech/Desktop/yujung/')
# os.getcwd()
# os.path
parser = argparse.ArgumentParser()
parser.add_argument('--folder1', type=str, default='0', required=False)
parser.add_argument('--folder2', type=str, default='1', required=False)
parser.add_argument('--batch_size1', type=int, required=True)
parser.add_argument('--batch_size2', type=int, required=True )

args = parser.parse_args()

# test셋과 train셋으로 데이터 나눔

renew_database = 1
# 파일 처리 - error예방
if (renew_database):
    if os.path.exists('train'):
        shutil.rmtree('train')
    if os.path.exists('test'):
        shutil.rmtree('test')
    if not os.path.exists('train'):
        os.mkdir('train')
    if not os.path.exists('test'):
        os.mkdir('test')

    rate = 0.1  # 분할 비율

    for root_dir in [args.folder1, args.folder2]:
        list_file = os.listdir(root_dir)
        picknumber = int(len(list_file) * rate)  # test이미지 개수
        test_pic = random.sample(list_file, picknumber)  # test중애서 랜덤으로 뽑기
        os.mkdir('train/' + root_dir)
        os.mkdir('test/' + root_dir)
        for item in list_file:
            if item in test_pic:
                shutil.copy(root_dir + '/' + item, 'test/' + root_dir + '/' + item)
            else:  # 나머지는 train
                shutil.copy(root_dir + '/' + item, 'train/' + root_dir + '/' + item)


train_dir = os.path.join('/home/caitech/Desktop/yujung/train/')
test_dir = os.path.join('/home/caitech/Desktop/yujung/test/')

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

IMAGE_SIZE = 224

#데이터를 읽고 이미지를 한번에 넣은 개수_train
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),batch_size=args.batch_size1, class_mode='binary')

#데이터를 읽고 이미지를 한번에 넣은 개수_validation
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),batch_size = args.batch_size2, class_mode='binary')

#vgg model생성
pre_trained_vgg = VGG16(weights = 'imagenet',include_top = False,input_shape = (224,224,3))

for layer in pre_trained_vgg.layers:
    layer.trainable = False


vgg_model = tf.keras.Sequential()
vgg_model.add(pre_trained_vgg)
vgg_model.add(layers.Flatten())
vgg_model.add(layers.Dense(1024,activation = 'relu'))
vgg_model.add(Dropout(0.5))
vgg_model.add(layers.Dense(512,activation = 'relu'))
vgg_model.add(Dropout(0.5))
vgg_model.add(layers.Dense(256,activation = 'relu'))
vgg_model.add(Dropout(0.5))
vgg_model.add(layers.Dense(2,activation = 'softmax'))

#vgg_model.summary()

early_stopping = EarlyStopping(patience = 30)

vgg_model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


history_vgg = vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
        epochs=100,
        validation_data=test_generator,
        validation_steps= math.ceil(test_generator.n / test_generator.batch_size),
        callbacks = [early_stopping]
        )


scores = vgg_model.evaluate_generator(test_generator, steps=math.ceil(test_generator.n / test_generator.batch_size))

# with open('vgg_history.xlsx','w') as f:
#     f.write(history_vgg_df)

history_vgg_df = pd.DataFrame(history_vgg.history)
scores_df = pd.DataFrame(scores)

history_vgg_df.to_csv('./vgg_history.csv',mode = 'a')
scores_df.to_csv('./vgg_scores.csv',mode='a')


#Confution Matrix and Classification Report
#Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
Y_pred = vgg_model.predict_generator(test_generator, math.ceil(test_generator.n / test_generator.batch_size))
y_pred = np.argmax(Y_pred, axis=1)

# len(y_pred)

# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
# print('Classification Report')

#target_names = ['Cats', 'Dogs', 'Horse']
target_names = ['0', '1']
#print(classification_report(test_generator.classes, y_pred, target_names=target_names))

vgg_report = classification_report(test_generator.classes, y_pred, target_names=target_names,output_dict=True)

vgg_report_df = pd.DataFrame(vgg_report).transpose()

vgg_report_df.to_csv('./vgg_report.csv',mode = 'a')

# from sklearn import metrics
# print(metrics.classification_report(test_generator.classes, y_pred))
# type(vgg_report)
# vgg_report.values()
# vgg_report.key('0')
# vgg_report.items()

print('finish!')

