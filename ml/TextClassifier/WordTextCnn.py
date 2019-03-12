#coding:utf-8

import tensorflow as tf
import pandas as pd
import os, pickle
from collections import defaultdict


class WordTextCNN():
    def __init__(self, max_sequence_lenth, vocab_size, embedding_size, class_num, filter_sizes, filter_num, l2_reg=0.0, loss_alpha=0.25, loss_gamma=2):
        self.max_sequence_lenth = max_sequence_lenth
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.word_conv_filters = [2,3,4]
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma

        with tf.name_scope('palceholder'):
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.input_x = tf.placeholder(tf.int32, [None, max_sequence_lenth], name='input_x')
            self.input_y = tf.placeholder(tf.int32, [None, class_num], name='input_y')

        self.l2_loss = tf.constant(0.0)

        embedding_layer = self.__word2vec()
        word_conv_layer = self.__word_conv_pool_level(embedding_layer)
        sent_conv_layer = self.__sent_conv_pool_level(word_conv_layer)
        out = self.__classifier(sent_conv_layer)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            loss = -tf.reduce_mean(tf.cast(self.input_y, tf.float32) * tf.cast(self.loss_alpha, tf.float32) *
                                        tf.pow((1 - tf.clip_by_value(out, 1e-10, 1.0)), self.loss_gamma) *
                                        tf.log(tf.clip_by_value(out, 1e-10, 1.0)))
            self.loss = loss + l2_reg * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            predictions = tf.argmax(out, 1, name='prediction')
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def __word2vec(self):
        with tf.name_scope('embedding'):
            embedding_mat = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size]))
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            word_embedded = tf.expand_dims(word_embedded, -1)
        return word_embedded


    def __word_conv_pool_level(self, word2vec):
        pool_out = list()
        for filter_size in self.word_conv_filters:
            with tf.name_scope('word_conv-maxpool-{}'.format(filter_size)):
                W = tf.Variable(tf.truncated_normal([filter_size, self.embedding_size, 1, self.filter_num//2], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.filter_num//2]), name='b')
                conv = tf.nn.conv2d(
                    word2vec, W,
                    strides=[1,1,1,1],
                    padding='SAME',
                    name='conv'
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, filter_size, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='pool',
                )
                pool_out.append(pooled)

        return pool_out


    def __sent_conv_pool_level(self, word_conv):
        pool_out = list()
        for i, ench_word_level in enumerate(word_conv):
            for filter_size in self.filter_sizes:
                with tf.name_scope('sent_conv-maxpool-{}'.format(filter_size)):
                    W = tf.Variable(tf.truncated_normal([filter_size, 1, self.filter_num//2, self.filter_num], stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name='b')
                    conv = tf.nn.conv2d(
                        ench_word_level, W,
                        strides=[1,1,1,1],
                        padding='SAME',
                        name='conv'
                    )
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, (self.max_sequence_lenth-self.word_conv_filters[i]+1)-filter_size+1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name='pool',
                    )
                    pool_out.append(pooled)
        pool_out = tf.concat(pool_out, 3)
        pool_out_flat = tf.reshape(pool_out, [-1, len(self.filter_sizes)*self.filter_num])

        with tf.name_scope('dropout'):
            pool_dropout = tf.nn.dropout(pool_out_flat, self.dropout_keep_prob)

        return pool_dropout


    def __classifier(self, sent_conv):
        with tf.name_scope('out'):
            W = tf.get_variable(
                "W",
                shape=[len(self.filter_sizes)*self.filter_num, self.class_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            out = tf.nn.xw_plus_b(sent_conv, W, b, name="scores")
            out = tf.nn.softmax(out)
        return out


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
        print('train/val/test = {}/{}/{}'.format(len(self.train_data), len(self.val_data), len(self.test_data)))


    def get_dict(self, path):
        if os.path.exists(os.path.join(path, 'vocab.dic')):
            with open(os.path.join(path, 'vocab.dic'), 'rb') as fp:
                self.vocab = pickle.load(fp)
        else:
            all_text = self.data['text'].tolist()
            tmp = defaultdict(lambda : 0)

            for text in all_text:
                for word in text.split(' '):
                    tmp[word] += 1

            tmp = [(k, v) for k, v in tmp.items()]
            tmp.sort(key=lambda x:x[1], reverse=True)

            tmp = [x[0] for x in tmp]

            self.vocab = dict()
            for index, word in enumerate(tmp):
                self.vocab[word] = index+1

            print('writing dic...')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, 'vocab.dic'), 'wb') as fp:
                pickle.dump(self.vocab, fp)

    def process_data(self):
        def process_text(doc):
            tmp = map(lambda x: self.vocab[x] if x in self.vocab else 0, doc.split(' '))
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