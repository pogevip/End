#coding:utf-8

import tensorflow as tf
from tensorflow.contrib import layers


class TextCNN():
    def __init__(self, max_sequence_lenth, vocab_size, embedding_size, class_num, filter_sizes, filter_num, l2_reg=0.0, loss_alpha=0.25, loss_gamma=2):
        self.max_sequence_lenth = max_sequence_lenth
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
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
        conv_pool_layer = self.__conv_pool_level(embedding_layer)
        out = self.__classifier(conv_pool_layer)

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


    def __conv_pool_level(self, word2vec):
        pool_out = list()
        for filter_size in self.filter_sizes:
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                W = tf.Variable(tf.truncated_normal([filter_size, self.embedding_size, 1, self.filter_num], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name='b')
                conv = tf.nn.conv2d(
                    word2vec, W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv'
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_sequence_lenth-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool',
                )
                pool_out.append(pooled)
        pool_out = tf.concat(pool_out, 3)
        pool_out_flat = tf.reshape(pool_out, [-1, len(self.filter_sizes)*self.filter_num])

        with tf.name_scope('dropout'):
            pool_dropout = tf.nn.dropout(pool_out_flat, self.dropout_keep_prob)

        return pool_dropout


    def __classifier(self, conv_pool):
        with tf.name_scope('out'):
            W = tf.get_variable(
                "W",
                shape=[len(self.filter_sizes)*self.filter_num, self.class_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            out = tf.nn.xw_plus_b(conv_pool, W, b, name="scores")
            out = tf.nn.softmax(out)
        return out