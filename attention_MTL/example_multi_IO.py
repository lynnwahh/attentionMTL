#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: example_multi_IO.py 
@time: 2021/08/16
@software: PyCharm 
'''
from keras import Input, Model
from keras.layers import Dense, Concatenate
import numpy as np
from keras.utils import plot_model
from numpy import random as rd
import pylab

def plot_samples(x, y1, y2, y3):
    pylab.figure(figsize=(3, 1.5))
    pylab.scatter(x[:, 0], y1)
    pylab.scatter(x[:, 0], y2)
    pylab.scatter(x[:, 0], y3)
    pylab.show()

samples_n = 300
samples_dim_01 = 2
samples_dim_02 = 2
# 样本数据
x1 = rd.rand(samples_n, samples_dim_01)
x2 = rd.rand(samples_n, samples_dim_02)
y_1 = []
y_2 = []
y_3 = []
for x11, x22 in zip(x1, x2):
    y_1.append(np.sum(x11) + np.sum(x22))
    y_2.append(np.max([np.max(x11), np.max(x22)]))
    y_3.append(np.min([np.min(x11), np.min(x22)]))
y_1 = np.array(y_1)
y_1 = np.expand_dims(y_1, axis=1)
y_2 = np.array(y_2)
y_2 = np.expand_dims(y_2, axis=1)
y_3 = np.array(y_3)
y_3 = np.expand_dims(y_3, axis=1)

# 输入层
inputs_01 = Input((samples_dim_01,), name='input_1')
inputs_02 = Input((samples_dim_02,), name='input_2')
# 全连接层
dense_01 = Dense(units=3, name="dense_01", activation='softmax')(inputs_01)
dense_011 = Dense(units=3, name="dense_011", activation='softmax')(dense_01)
dense_02 = Dense(units=6, name="dense_02", activation='softmax')(inputs_02)
# 加入合并层
merge = Concatenate()([dense_011, dense_02])
# 分成两类输出 --- 输出01
output_01 = Dense(units=6, activation="relu", name='output01')(merge)
output_011 = Dense(units=1, activation=None, name='output011')(output_01)
# 分成两类输出 --- 输出02
output_02 = Dense(units=1, activation=None, name='output02')(merge)
# 分成两类输出 --- 输出03
output_03 = Dense(units=1, activation=None, name='output03')(merge)
# 构造一个新模型
model = Model(inputs=[inputs_01, inputs_02], outputs=[output_011,
                                                      output_02,
                                                      output_03
                                                      ])
# 显示模型情况
import os
# os.environ["PATH"] += os.pathsep + '"C:\Program Files\Graphviz\bin'
#plot_model(model, show_shapes=True)
print(model.summary())
# # 编译
# model.compile(optimizer="adam", loss='mean_squared_error', loss_weights=[1,
#                                                                          0.8,
#                                                                          0.8
#                                                                          ])
# # 训练
# model.fit([x1, x2], [y_1,
#                      y_2,
#                      y_3
#                      ], epochs=50, batch_size=32, validation_split=0.1)

# 以下的方法可灵活设置
model.compile(optimizer='adam',
              loss={'output011': 'mean_squared_error',
                    'output02': 'mean_squared_error',
                    'output03': 'mean_squared_error'},
              loss_weights={'output011': 1,
                            'output02': 0.8,
                            'output03': 0.8})
model.fit({'input_1': x1,
           'input_2': x2},
          {'output011': y_1,
           'output02': y_2,
           'output03': y_3},
          epochs=50, batch_size=32, validation_split=0.1)

# 预测
test_x1 = rd.rand(1, 2)
test_x2 = rd.rand(1, 2)
test_y = model.predict(x=[test_x1, test_x2])
# 测试
print("测试结果：")
print("test_x1:", test_x1, "test_x2:", test_x2, "y:", test_y, np.sum(test_x1) + np.sum(test_x2))

plot_samples(x1, y_1, y_2, y_3)
plot_samples(test_x1, test_y[0], test_y[1], test_y[2])