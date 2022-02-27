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
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import models,layers, Input, optimizers, initializers, regularizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,  EarlyStopping
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop

os.chdir('/home/caitech/Desktop/yujung/')

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

IMAGE_SIZE = 299

#데이터를 읽고 이미지를 한번에 넣은 개수_train
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),batch_size=args.batch_size1, class_mode='binary')

#데이터를 읽고 이미지를 한번에 넣은 개수_validation
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),batch_size = args.batch_size2, class_mode='binary')

pre_inceptionV3 = InceptionV3(weights = 'imagenet',include_top=False, input_shape = (299,299,3))

v3_model = tf.keras.Sequential()
v3_model.add(pre_inceptionV3)
v3_model.add(layers.GlobalAveragePooling2D())
v3_model.add(layers.Dense(2,activation = 'softmax'))

for layer in pre_inceptionV3.layers:
    layer.trainable = False

early_stopping = EarlyStopping(patience=30)

v3_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              #optimizer = 'RMSprop',
              metrics=['accuracy'],
             )

history_inceptionv3= v3_model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
        epochs=100,
        validation_data=test_generator,
        validation_steps=math.ceil(test_generator.n / test_generator.batch_size),
        callbacks=[early_stopping]
        )

v3_scores = v3_model.evaluate_generator(test_generator, steps=math.ceil(test_generator.n / test_generator.batch_size))

v3_history_df = pd.DataFrame(history_inceptionv3.history)
v3_scores_df = pd.DataFrame(v3_scores)

v3_history_df.to_csv('./inception_history.csv', mode='a')
v3_scores_df.to_csv('./inception_scores.csv', mode='a')

Y_pred_v3 = v3_model.predict_generator(test_generator, steps=math.ceil(test_generator.n / test_generator.batch_size))
y_pred_v3 = np.argmax(Y_pred_v3, axis=1)

# len(y_pred)

# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
# print('Classification Report')

# target_names = ['Cats', 'Dogs', 'Horse']
target_names = ['0', '1']
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))

v3_report = classification_report(test_generator.classes, y_pred_v3, target_names=target_names, output_dict=True)

v3_report_df = pd.DataFrame(v3_report).transpose()

v3_report_df.to_csv('./inceptionv3_report.csv', mode = 'a')

# from sklearn import metrics
# print(metrics.classification_report(test_generator.classes, y_pred))
# type(vgg_report)
# vgg_report.values()
# vgg_report.key('0')
# vgg_report.items()

print('finish!')
