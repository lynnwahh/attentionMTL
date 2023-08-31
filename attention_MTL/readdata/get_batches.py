#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: get_batches.py 
@time: 2021/05/08
@software: PyCharm 
'''
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils import np_utils
from config import cfg
import eventlet

def get_dirs(data_dir):
    feature_dirs = []
    labels = []
    class_list = os.listdir(data_dir)
    class_list.sort()
    id = 0
    for cla in class_list:
        cla_dir = os.path.join(data_dir, cla)
        file_list = os.listdir(cla_dir)
        file_list.sort()
        for file in file_list:
            file_dir = os.path.join(cla_dir, file)
            feature_dirs.append(file_dir)
            labels.append(id)
        id = id + 1
    feature_dirs = np.asarray(feature_dirs)
    labels = np.asarray(labels)

    return feature_dirs, labels

def get_MTLdirs(data_dir):
    feature_dirs = []
    infiltration = []
    growth = []
    class_list = os.listdir(data_dir)
    class_list.sort()
    for cla in class_list:
        (left, right) = cla.split('_')
        cla_dir = os.path.join(data_dir, cla)
        file_list = os.listdir(cla_dir)
        file_list.sort()
        for file in file_list:
            file_dir = os.path.join(cla_dir, file)
            feature_dirs.append(file_dir)
            growth.append(left)
            infiltration.append(right)
    feature_dirs = np.asarray(feature_dirs)
    growth = np.asarray(growth)
    infiltration = np.asarray(infiltration)

    return feature_dirs, growth, infiltration


def get_id_reg_dirs(data_dir, reg_dir, reg_value):
    ids = []
    regs = []
    feature_dirs = []

    df = pd.read_csv(reg_dir)
    id_list = os.listdir(data_dir)
    id_list.sort(key=int)
    for id in id_list:
        id_dir = os.path.join(data_dir, id)
        files = os.listdir(id_dir)
        files.sort()
        reg = df[df['id'] == int(id)][reg_value]
        for file in files:
            file_data_dir = os.path.join(id_dir, file)
            feature_dirs.append(file_data_dir)
            ids.append(id)
            regs.append(reg)
    feature_dirs = np.asarray(feature_dirs)
    ids = np.asarray(ids)
    if len(reg_value) == 1:
        regs = np.asarray(regs).reshape([ids.shape[0]])
    elif len(reg_value) <= 0:
        print("no regression value!")
    else:
        regs = np.asarray(regs).reshape([ids.shape[0], len(reg_value)])

    return feature_dirs, ids, regs

def get_id_dirs(data_dir):
    ids = []
    feature_dirs = []

    id_list = os.listdir(data_dir)
    id_list.sort(key=int)
    for id in id_list:
        id_dir = os.path.join(data_dir, id)
        files = os.listdir(id_dir)
        files.sort()
        for file in files:
            file_data_dir = os.path.join(id_dir, file)
            feature_dirs.append(file_data_dir)
            ids.append(id)
    feature_dirs = np.asarray(feature_dirs)
    ids = np.asarray(ids)

    return feature_dirs, ids

def split_dirs(data_dir, num_classes=False, reg_dir=False):
    if reg_dir:
        feature_dirs, ids = get_id_dirs(data_dir)
        df = pd.read_csv(reg_dir)
        ids_df = pd.DataFrame(ids, columns=['id'], dtype=int)
        relations = pd.merge(ids_df, df, how='left', on='id')
        X_train, X_valid, reg_train, reg_valid = train_test_split(feature_dirs, relations, test_size=0.3, random_state=20)
        return X_train, X_valid, reg_train, reg_valid

    elif num_classes:
        feature_dirs, labels = get_dirs(data_dir)
        X_train, X_valid, y_train, y_valid = train_test_split(feature_dirs, labels, test_size=0.3, random_state=20)
        Y_train = np_utils.to_categorical(y_train, num_classes)
        Y_valid = np_utils.to_categorical(y_valid, num_classes)
        return X_train, X_valid, Y_train, Y_valid

    else:
        print("Neighther classification nor regression input")
        return None


def kfold_split_dirs(data_dir, k, num_growth, num_infiltration):
    df = pd.read_csv(data_dir)
    feature_dirs = np.asarray(df['img_dir'])
    growth = np.asarray(df['growth'])
    infiltration = np.asarray(df['infiltration'])
    
    X_train_list = []
    X_valid_list = []
    Y_train_growth_list = []
    Y_valid_growth_list = []
    Y_train_infiltration_list = []
    Y_valid_infiltration_list = []

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=20)  # 各个类别的比例大致和完整数据集中相同
    for train, test in skf.split(feature_dirs, growth, infiltration):
        print("%s-fold split, ：%s %s" % (k, train.shape, test.shape))
        # print("TRAIN_INDEX:", train, "TEST_INDEX:", test)
        X_train_list.append(feature_dirs[train])
        X_valid_list.append(feature_dirs[test])

        y_train = growth[train]
        y_valid = growth[test]
        Y_train = np_utils.to_categorical(y_train, num_growth)
        Y_valid = np_utils.to_categorical(y_valid, num_growth)
        Y_train_growth_list.append(Y_train)
        Y_valid_growth_list.append(Y_valid)
        
        y_train = infiltration[train]
        y_valid = infiltration[test]
        Y_train = np_utils.to_categorical(y_train, num_infiltration)
        Y_valid = np_utils.to_categorical(y_valid, num_infiltration)
        Y_train_infiltration_list.append(Y_train)
        Y_valid_infiltration_list.append(Y_valid)

    X_train = np.asarray(X_train_list, dtype=object)
    X_valid = np.asarray(X_valid_list, dtype=object)
    Y_train_growth = np.asarray(Y_train_growth_list, dtype=object)
    Y_valid_growth = np.asarray(Y_valid_growth_list, dtype=object)
    Y_train_infiltration = np.asarray(Y_train_infiltration_list, dtype=object)
    Y_valid_infiltration = np.asarray(Y_valid_infiltration_list, dtype=object)

    return X_train, X_valid, Y_train_growth, Y_valid_growth, Y_train_infiltration, Y_valid_infiltration

def read_oneimg(dir, img_rows=224, img_cols=224, color_type=3, normalize=True):
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(img, (img_rows, img_cols),
                         interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32)
    if (np.max(resized) - np.min(resized)) != 0 and (np.max(resized) - np.min(resized)) != None:
        resized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized)) * 255

    return resized

def get_imgs(dirs, img_rows=224, img_cols=224, color_type=3, normalize=True):
    imgs = []
    for dir in dirs:
        img = read_oneimg(dir)
        imgs.append(img)

    return np.array(imgs).reshape(len(dirs), img_rows, img_cols, color_type)

def get_batch(X, Y1, Y2, batch_size, is_aug=True):

    img_rows, img_cols = cfg.img_rows, cfg.img_cols
    color_type = cfg.color_type
    while 1:
        for i in range(0, len(X), batch_size):
            x = get_imgs(X[i:i+batch_size], img_rows, img_cols, color_type)
            y1 = Y1[i:i+batch_size]
            y2 = Y2[i:i+batch_size]
            #y = [y1, y2]
            #print("batch", i)
            #print("count from %d to %d, feature shape %s, label shape %s, multi task %s"
                  #%(i, i+batch_size, x.shape, y1.shape, len(y)))
            #if is_aug:
             #   x, y = img_augmentation(x, y)
            #yield({'input':x}, {'output':y})
            yield ({
                'img_input':x,
                'growth_true':y1,
                'infiltration_true':y2
            },
            None)




if __name__ == "__main__":
    data_dir = cfg.data_dir
    re_dir = cfg.csv_dir
    
    feature_dirs, growth, infiltration = get_MTLdirs(data_dir)
    df = pd.DataFrame({
        'img_dir':feature_dirs,
        'growth':growth,
        'infiltration':infiltration
    })
    df.to_csv(re_dir)
 

