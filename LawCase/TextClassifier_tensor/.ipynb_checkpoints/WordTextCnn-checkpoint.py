#coding:utf-8

import tensorflow as tf
from keras import callbacks
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, GRU, Dense, Dropout
import keras.backend as K
import numpy as np
import pandas as pd
import os, pickle
from collections import defaultdict
import random


class WTCTrainDataGenerator():
    def __init__(self, src_path):
        self.code2label = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        print('load data...')
        self.data = pd.read_csv(src_path)


    def preprocess_data(self):

        train_data = self.data[self.data['train_val_test'] == 1]
        val_data = self.data[self.data['train_val_test'] == 2]
        test_data = self.data[self.data['train_val_test'] == 3]

        train_data.dropna(how='any', inplace=True)
        val_data.dropna(how='any', inplace=True)
        test_data.dropna(how='any', inplace=True)

        self.train_data = train_data.sample(frac=1).reset_index(drop=True)
        self.val_data = val_data
        self.test_data = test_data

        print('load finished.')
        print('train/test = {}/{}'.format(len(self.train_data), len(self.test_data)))


    def get_dict(self, path):
        if os.path.exists(os.path.join(path, 'word.dic')):
            with open(os.path.join(path, 'word.dic'), 'rb') as fp:
                self.vocab = pickle.load(fp)
        else:
            all_text = self.data['text'].tolist()
            tmp = defaultdict(lambda : 0)

            for text in all_text:
                for word in text.split(' '):
                    tmp[word] += 1

            tmp = [(k, v) for k, v in tmp.items()]
            tmp.sort(key=lambda x:x[1], reverse=True)

            tmp = filter(lambda x: x[1] > 10, tmp)
            tmp = [x[0] for x in tmp]

            self.vocab = dict()
            for index, word in enumerate(tmp):
                self.vocab[word] = index

            print('writing dic...')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, 'word.dic'), 'wb') as fp:
                pickle.dump(self.vocab, fp)

    def process_data(self):
        def process_text(doc):
            tmp = map(lambda x: self.vocab[x] if x in self.vocab else None, doc.split(' '))
            res = list(filter(lambda x: x, tmp))
            return res
        print('train data')
        # 训练集
        self.x_train = self.train_data['text'].apply(process_text).tolist()
        train_y = self.train_data['cls'].tolist()

        self.code2label = {}
        for index, item in enumerate(list(set(train_y))):
            self.code2label[item] = index

        self.y_train = list(map(lambda x: self.code2label[x], train_y))

        print('val data')
        # Test set
        self.x_val = self.val_data['text'].apply(process_text).tolist()
        self.y_val = list(map(lambda x: self.code2label[x], self.val_data['cls'].tolist()))

        print('test data')
        # Test set
        self.x_test = self.test_data['text'].apply(process_text).tolist()
        self.y_test = list(map(lambda x: self.code2label[x], self.test_data['cls'].tolist()))


    def out(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        print('processing...')
        self.process_data()

        print('out cls_map...')
        with open(os.path.join(path, 'cls_label.dic'), 'wb') as fp:
            pickle.dump(self.code2label, fp)

        print('out train data...')
        with open(os.path.join(path, 'train'), 'wb') as fp:
            pickle.dump((self.x_train, self.y_train), fp)

        print('out val data...')
        with open(os.path.join(path, 'val'), 'wb') as fp:
            pickle.dump((self.x_val, self.y_val), fp)

        print('out test data...')
        with open(os.path.join(path, 'test'), 'wb') as fp:
            pickle.dump((self.x_test, self.y_test), fp)


class DataReader():
    def __init__(self, path, batch_size=0):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        self.x = data[0]
        self.y = data[1]
        self.batch_size = batch_size

    def read(self):
        def stuff_doc(doc, max_length=2500):
            if len(doc) > max_length:
                start_index = (len(doc) - max_length - 1) // 2 + 1
                doc = doc[start_index: start_index + max_length]
            else:
                doc.extend([0] * (max_length - len(doc)))
            return doc

        def gen_one_hot(label):
            label = np.array(label)
            return (np.arange(10) == label[:, None]).astype(np.int32)

        if self.batch_size == 0:
            x = np.array(list(map(lambda doc: stuff_doc(doc), self.x)))
            y = gen_one_hot(self.y)
            return x, y
        else:
            data_size = len(self.x)
            num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
            shuffle_indices = list(range(data_size))
            random.shuffle(shuffle_indices)

            shuffled_x = [self.x[i] for i in shuffle_indices]
            shuffled_y = [self.y[i] for i in shuffle_indices]


            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, data_size)

                x_tmp = shuffled_x[start_index:end_index]
                y_tmp = shuffled_y[start_index:end_index]

                x = np.array(list(map(lambda doc: stuff_doc(doc), x_tmp)))
                y = gen_one_hot(y_tmp)

                yield x, y


class WordTextCnn():
    def __init__(self, maxlen, max_features, embedding_dims, class_num, filters = 256):
        # Inputs
        seq = Input(shape=[maxlen], name='x_seq')

        # Embedding layers
        emb = Embedding(max_features, embedding_dims)(seq)

        # conv layers
        words_infos = []
        filter_sizes1 = [2, 3, 4, 5]
        for fsz in filter_sizes1:
            conv1 = Conv1D(filters, kernel_size=fsz, activation='relu')(emb)
            pool1 = MaxPooling1D(fsz, strides=fsz)(conv1)
            words_infos.append(pool1)

        convs = []
        filter_sizes2 = [3, 5, 8]
        for words_info in words_infos:
            for fsz in filter_sizes2:
                conv2 = Conv1D(filters, kernel_size=(fsz, filters), activation='relu')(words_info)
                pool2 = MaxPooling1D(fsz, strides=fsz)(conv2)
                pool2 = Flatten()(pool2)
                convs.append(pool2)
        merge = K.concatenate(convs, axis=1)

        out = Dropout(0.5)(merge)
        output = Dense(32, activation='relu')(out)

        output = Dense(units=class_num, activation='sigmoid')(output)

        self.model = Model([seq], output)

        def mycrossentropy(y_true, y_pred, loss_alpha=0.25, loss_gamma=2):
            loss = -tf.reduce_mean(tf.cast(y_true, tf.float32) * tf.cast(loss_alpha, tf.float32) *
                                   tf.pow((1 - tf.clip_by_value(y_pred, 1e-10, 1.0)), loss_gamma) *
                                   tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
            return loss

        self.model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, train_data_generator, val_x, val_y, out_dir, batch_size = 128, epochs = 10):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filepath = os.path.join(out_dir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        print("training model")
        history = self.model.fit(train_data_generator, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y), verbose=2, shuffle=True, callbacks=callbacks_list)
        accy = history.history['acc']
        np_accy = np.array(accy)
        np.savetxt(os.path.join(out_dir, 'acc.txt'), np_accy)

    def test(self, x, y):
        print("pridicting...")
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('test_loss:%f,accuracy: %f' % (loss, acc))

        print("saving textcnnmodel")
        self.model.save('./predictor/model/%s_cnn_large.h5')