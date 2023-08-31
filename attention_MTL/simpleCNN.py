#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: simpleCNN.py 
@time: 2021/08/13
@software: PyCharm 
'''
import keras, os
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda, Layer
from keras.initializers import Constant
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.metrics import categorical_accuracy
from config import cfg
from readdata.get_batches import kfold_split_dirs, get_batch
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICE


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

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

def build_model(img_rows, img_cols, color_type, num_growth, num_infiltration):
    filter_size = (5, 5)
    maxpool_size = (2, 2)
    dr = 0.3

    inputs = Input(shape=(img_rows, img_cols, color_type), name='main_input')

    main_branch = Conv2D(16, kernel_size=filter_size, padding="same")(inputs)
    main_branch = Activation("relu")(main_branch)
    main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
    main_branch = Dropout(dr)(main_branch)

    main_branch = Conv2D(8, kernel_size=filter_size, padding="same")(main_branch)
    main_branch = Activation("relu")(main_branch)
    main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
    main_branch = Dropout(dr)(main_branch)

    main_branch = Flatten()(main_branch)
    main_branch = Dense(32)(main_branch)
    main_branch = Activation('relu')(main_branch)
    main_branch = Dropout(dr)(main_branch)

    growth_branch = Dense(num_growth, activation='softmax', name='growth_output')(main_branch)
    infiltration_branch = Dense(num_infiltration, activation='softmax', name='infiltration_output')(main_branch)

    model = Model(inputs=inputs,
                  outputs=[growth_branch, infiltration_branch])
    print("prediction model")
    model.summary()

    # uncertainty model
    growth_true = Input(shape=(num_growth, ), name='growth_true')
    infiltration_true = Input(shape=(num_infiltration, ), name='infiltration_true')
    out = CustomMultiLossLayer(nb_outputs=2)([growth_true, infiltration_true,
                                              growth_branch, infiltration_branch])
    model = Model(inputs=[inputs, growth_true, infiltration_true],
                  outputs=[out, growth_branch, infiltration_branch])
    model.add_metric(categorical_accuracy(growth_true, growth_branch),
                     name='growth_acc')
    model.add_metric(categorical_accuracy(infiltration_true, infiltration_branch),
                     name='infiltration_acc')
    print("uncertainty weighing loss model")
    model.summary()


    sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  #loss={'growth_output': 'categorical_crossentropy', 'infiltration_output': 'categorical_crossentropy'},
                  #loss_weights={'growth_output': 0.5, 'infiltration_output': 0.5},
                  #metrics={'growth_output': 'categorical_accuracy', 'infiltration_output': 'categorical_accuracy'},
                  loss=None,
                  )
    #assert len(model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
    #assert len(model.losses) == 1

    return model

if __name__ == "__main__":
    model_name = cfg.model_name
    to_dir = cfg.OUTPUT_DIR
    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_growth = 7
    num_infiltration = 3
    batch_size = cfg.BATCH_SIZE
    nb_epoch = cfg.NB_EPOCHS
    kfold = cfg.KFOLD
    print(kfold, 'fold')
    data_dir = 'D:/Data/imageprocessing/MTLmetadata.csv'

    X_train, X_valid, Y_train_growth, Y_valid_growth, \
    Y_train_infiltration, Y_valid_infiltration = \
        kfold_split_dirs(data_dir, kfold, num_growth, num_infiltration)

    for i in range(kfold):
        print("***********************KFOLD-%s***********************" % i)
        train_steps = X_train[i].shape[0] // batch_size + 1
        valid_steps = X_valid[i].shape[0] // batch_size + 1

        from keras.callbacks import ModelCheckpoint, TensorBoard
        ck_dir = os.path.join("./checkpoint/" + model_name + str(kfold) + 'fold', str(i))
        os.makedirs(ck_dir, exist_ok=True)
        checkpoint = ModelCheckpoint(os.path.join(ck_dir, 'model_{epoch:03d}_{val_loss:.2f}.hdf5'),
                                     monitor='val_loss',
                                     verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]
        '''
        #add tensorboard will be slower
        log_dir = os.path.join("./log/"+model_name+str(kfold)+'fold', str(i))
        os.makedirs(log_dir, exist_ok=True)
        log = TensorBoard(log_dir=log_dir, write_images=1, histogram_freq=1, update_freq='batch')
        callbacks_list = [checkpoint, log]
        '''

        model = build_model(img_rows, img_cols, channel, num_growth, num_infiltration)
        history = model.fit_generator(
            generator=get_batch(X_train[i], Y_train_growth[i], Y_train_infiltration[i], batch_size),
                  epochs=nb_epoch, steps_per_epoch=train_steps, verbose=1,
             validation_data=get_batch(X_valid[i], Y_valid_growth[i], Y_valid_infiltration[i], batch_size),
                  callbacks=callbacks_list, max_queue_size=10, validation_steps=valid_steps,
                  workers=cfg.WORKERS, use_multiprocessing=cfg.MULTI_PRO
                  )
