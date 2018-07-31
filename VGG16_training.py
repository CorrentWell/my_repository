# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:09:17 2018

@author: yg2018
"""

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import time
from matplotlib import pyplot as plt
import tensorflow as tf

# 分類するクラス
classes = ["BG", "HCC"]
nb_classes = len(classes)

img_width, img_height = 200, 200

# トレーニング用とバリデーション用の画像格納先
train_data_dir = 'for_VGG16/train'
validation_data_dir = 'for_VGG16/validation'

# 今回はトレーニング用に880枚、バリデーション用に220枚の画像を用意した。
nb_train_samples = 880
nb_validation_samples = 220

batch_size = 20 #テストデータ数を割り切れる数にするべき
nb_epoch = 50


result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def vgg_model_maker():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    return model


def image_generator():
    """ ディレクトリ内の画像を読み込んでトレーニングデータとバリデーションデータの作成 """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    return (train_generator, validation_generator)


if __name__ == '__main__':
    start = time.time()

    # モデル作成
    vgg_model = vgg_model_maker()

    # 最後のconv層の直前までの層をfreeze
    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    # 多クラス分類を指定
    """
    vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])
    """
    vgg_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

    # 画像のジェネレータ生成
    train_generator, validation_generator = image_generator()

    # Fine-tuning
    history = vgg_model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
    
    x = range(nb_epoch)
    for h in history:
        plt.plot(x, history.history['acc'], label="test")
        plt.plot(x, history.history['val_acc'], label="validation")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    vgg_model.save_weights(os.path.join(result_dir, 'finetuning.h5'))

    process_time = (time.time() - start) / 60
    print(u'学習終了。かかった時間は', process_time, u'分です。')