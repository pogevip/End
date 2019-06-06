# coding:utf-8

import os, math
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from Rank.data_helper import load_data, MAX_LEN
import numpy as np




def test(data_path, model_path):
    print('load data')
    test_data = load_data(os.path.join(data_path, 'test.pkl'))

    X1 = test_data['X1']
    X1 = np.array(list(map(lambda doc: (doc * math.ceil(MAX_LEN / len(doc)))[:MAX_LEN], X1)))
    X2 = test_data['X2']
    X2 = np.array(list(map(lambda doc: (doc * math.ceil(MAX_LEN / len(doc)))[:MAX_LEN], X2)))
    y_true = test_data['Y']

    print('load model')

    def my_loss(y_true, y_pred):
        return K.mean(K.pow(K.log(y_pred+1)-K.log(y_true+1), 2))
    model = load_model(os.path.join(model_path, 'model.h5'), {'my_loss': my_loss})

    print('evaluate...')
    loss = model.evaluate([X1, X2], y_true, batch_size=256)

    print(loss)

    return loss



if __name__ == '__main__':
    dir = 'data/trainSet/rank/'
    cls = '9771'
    model_op = 'cnn'

    model_path = os.path.join(dir, 'training', cls, model_op)
    data_path = os.path.join(dir, 'train_file', cls)

    loss = test(data_path, model_path)