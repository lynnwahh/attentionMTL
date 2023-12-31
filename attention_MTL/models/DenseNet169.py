#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: DenseNet169.py 
@time: 2021/08/16
@software: PyCharm 
'''

import warnings
warnings.filterwarnings("ignore")

from keras.optimizers import SGD
from keras.layers import Input, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
from custom_layers.scale_layer import Scale
import os
from keras.utils import multi_gpu_model


def densenet169_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5,
                      dropout_rate=0.0, weight_decay=1e-4, num_growth=7, num_infiltration=3):
    '''
    DenseNet 169 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    K.clear_session()  # 清除之前的模型，省得压满内存
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    #if K.image_dim_ordering() == 'tf':  #channels_last=tf
    if K.image_data_format() == 'channels_last':
        concat_axis = 3
        img_input = Input(shape=(img_rows, img_cols, 3), name='img_input')
    else:
        concat_axis = 1
        img_input = Input(shape=(3, img_rows, img_cols), name='img_input')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 32, 32]  # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')
    '''
    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'imagenet_models/densenet169_weights_th.h5'
    else:
    '''
    # Use pre-trained weights for Tensorflow backend
    from config import cfg
    weights_path = cfg.pretrain_path

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    growth_branch = Dense(num_growth, activation='softmax',
                          name='growth_output')(x_newfc)
    infiltration_branch = Dense(num_infiltration, activation='softmax',
                                name='infiltration_output')(x_newfc)

    growth_true = Input(shape=(num_growth,), name='growth_true')
    infiltration_true = Input(shape=(num_infiltration,), name='infiltration_true')
    from MTLloss.uncertain_weigh_loss import CustomMultiLossLayer
    out = CustomMultiLossLayer(nb_outputs=2)([growth_true, infiltration_true,
                                              growth_branch, infiltration_branch])
    model = Model(inputs=[img_input, growth_true, infiltration_true],
                  outputs=[out, growth_branch, infiltration_branch])
    from keras.metrics import categorical_accuracy
    model.add_metric(categorical_accuracy(growth_true, growth_branch),
                     name='growth_acc')
    model.add_metric(categorical_accuracy(infiltration_true, infiltration_branch),
                     name='infiltration_acc')
    #print("uncertainty weighing loss model")
    #model.summary()
    #if multi_GPU > 1:
        #model = multi_gpu_model(model, multi_GPU)


    # Learning rate is changed to 0.001
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss=None)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        #concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,
                            #name='concat_' + str(stage) + '_' + str(branch))
        from keras.layers.merge import concatenate

        concat_feat = concatenate([concat_feat, x], axis=concat_axis,
                                  name='concat_' + str(stage) + '_' + str(branch))
        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

