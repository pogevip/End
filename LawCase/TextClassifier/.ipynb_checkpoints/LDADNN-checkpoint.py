#coding:utf-8

from keras.layers import Dense, Input
from keras.models import Model
import os
import numpy as np

MAX_LEN = 200
BATCH_SIZE = 256

def LdaDnn(input_size):
    # Inputs
    comment_seq = Input(shape=[input_size], name='x_seq')

    hidden = Dense(128, activation='relu')(comment_seq)

    output = Dense(units=10, activation='softmax')(hidden)

    model = Model([comment_seq], output)
    return model