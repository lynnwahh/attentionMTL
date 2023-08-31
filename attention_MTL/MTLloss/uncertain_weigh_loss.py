#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: uncertain_weigh_loss.py 
@time: 2021/08/15
@software: PyCharm 
'''
import numpy as np
import keras
from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K

def gen_data(N):
    X = np.random.randn(N, Q)
    w1 = 2.
    b1 = 8.
    sigma1 = 1e1  # ground truth
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    w2 = 3
    b2 = 3.
    sigma2 = 1e0  # ground truth
    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2

def gen_batch(X, Y1, Y2, batch_size):
    while 1:
        for i in range(0, len(X), batch_size):
            x = X[i:i+batch_size]
            y1 = Y1[i:i+batch_size]
            y2 = Y2[i:i+batch_size]
            yield ({
                'inp':x,
                'y1_true':y1,
                'y2_true':y2
            }, None)

# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def rg_loss(ys_true, ys_pred, delta=0.01):
        tilde = K.round(ys_pred * 20) / 20
        loss = keras.losses.MeanSquaredError(ys_true, tilde, from_logits=True) + delta * keras.losses.MeanSquaredError(
            ys_true, ys_pred, from_logits=True)
        return loss

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1) + self.rg_loss(ys_true, ys_pred)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

def build_model():
    inp = Input(batch_shape=(batch_size, Q,), name='inp')
    x = Dense(nb_features, activation='relu')(inp)
    y1_pred = Dense(D1)(x)
    y2_pred = Dense(D2)(x)
    y1_true = Input(batch_shape=(batch_size, D1,), name='y1_true')
    y2_true = Input(batch_shape=(batch_size, D2,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    model = Model(inputs=[inp, y1_true, y2_true], outputs=out)
    model.summary()
    model.compile(optimizer='adam', loss=None, metrics=['mse'])
    assert len(model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
    assert len(model.losses) == 1

    return model


if __name__ == "__main__":
    N = 5000
    nb_epoch = 5000
    batch_size = 20
    nb_features = 1024
    Q = 1
    D1 = 1  # first output
    D2 = 1  # second output

    X, Y1, Y2 = gen_data(N)
    model = build_model()
    model.summary()
    #hist = model.fit([X, Y1, Y2], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    hist = model.fit_generator(gen_batch(X, Y1, Y2, batch_size), nb_epoch=nb_epoch, steps_per_epoch=250)