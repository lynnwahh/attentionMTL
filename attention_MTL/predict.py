#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: predict.py 
@time: 2021/08/16
@software: PyCharm 
'''
from simpleCNN import build_model
from readdata.get_batches import kfold_split_dirs, read_oneimg
from keras.utils import np_utils

if __name__ == "__main__":
    model_path = './checkpoint/simpleCNN4fold/0/model_001_0.04.hdf5'
    num_growth = 7
    num_infiltration = 3
    img_rows, img_cols, channel = 224, 224, 3
    data_dir = 'D:/Data/imageprocessing/MTLmetadata.csv'
    kfold = 4

    model = build_model(img_rows, img_cols, channel, num_growth, num_infiltration)
    model.load_weights(model_path)
    X_train, X_valid, Y_train_growth, Y_valid_growth, \
    Y_train_infiltration, Y_valid_infiltration = \
        kfold_split_dirs(data_dir, kfold, num_growth, num_infiltration)
    x_test = read_oneimg(X_valid[0][0]).reshape(1, img_rows, img_cols, channel)
    y_true_growth = np_utils.to_categorical(Y_valid_growth[0][0], num_growth)
    y_true_infiltration = np_utils.to_categorical(Y_valid_infiltration[0][0], num_infiltration)
    pred = model.predict([x_test, y_true_growth, y_true_infiltration])
    print("finish")