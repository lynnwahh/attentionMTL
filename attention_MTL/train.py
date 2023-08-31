#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: train.py 
@time: 2021/08/16
@software: PyCharm 
'''
import os
from models import densenet169_model, densenet169_attention_model
from models.DenseNet169_attention import densenet169_attention_model
from config import cfg
from readdata.get_batches import kfold_split_dirs, get_batch
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICE

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

    X_train, X_valid, Y_train_growth, Y_valid_growth, \
    Y_train_infiltration, Y_valid_infiltration = \
        kfold_split_dirs(cfg.csv_dir, kfold, num_growth, num_infiltration)

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
        if  cfg.model_name=='D169_':
            model = densenet169_model(img_rows, img_cols)
        elif cfg.model_name=='D169_attention_':
            model = densenet169_attention_model(img_rows, img_cols)

        history = model.fit_generator(
            generator=get_batch(X_train[i], Y_train_growth[i], Y_train_infiltration[i], batch_size),
                  epochs=nb_epoch, steps_per_epoch=train_steps, verbose=1,
             validation_data=get_batch(X_valid[i], Y_valid_growth[i], Y_valid_infiltration[i], batch_size),
                  callbacks=callbacks_list, max_queue_size=10, validation_steps=valid_steps,
                  workers=cfg.WORKERS, use_multiprocessing=cfg.MULTI_PRO
                  )