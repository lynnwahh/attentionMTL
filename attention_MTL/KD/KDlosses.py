#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: KDlosses.py 
@time: 2021/11/18
@software: PyCharm 
'''
import tensorflow as tf
import keras.backend as K
import random
import numpy as np

# extended softmax
def es_loss(student_logits, teacher_logits, temperature=0.5):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    es_loss = tf.compat.v1.losses.softmax_cross_entropy(
        teacher_probs, student_logits / temperature, temperature**2)
    return es_loss


def esce_loss(student_logits, teacher_logits,
                true_labels, temperature,
                alpha, beta):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    es_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature,
        from_logits=True)

    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        true_labels, student_logits, from_logits=True)

    total_loss = (alpha * es_loss) + (beta * ce_loss)
    return total_loss / (alpha + beta)


def unlabel_loss(y_true, y_pred, margin=0.4):
    total_lenght = y_pred.shape.as_list()[-1]
    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + margin
    loss = K.maximum(basic_loss, 0.0)

    return loss


def create_triple(x_labeled, y_labeled, x_train, y_train):
    x_anchors = []
    x_positives = []
    x_negatives = []
    for i in range(0, y_train.shape[0]):
        #random_index = random.randint(0, x_labeled.shape[0] - 1)
        indices_for_anchor = np.squeeze(np.where(y_labeled == i))
        x_anchor = np.mean(x_labeled[indices_for_anchor])

        indices_for_pos = np.squeeze(np.where(y_train == i))
        x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]


        indices_for_neg = np.squeeze(np.where(y_train != i))
        x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        x_anchors.append(x_anchor)
        x_positives.append(x_positive)
        x_negatives.append(x_negative)

    return np.array(x_anchors), np.array(x_positives), np.array(x_negatives)


if __name__ == "__main__":
