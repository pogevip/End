#coding:utf-8

import tensorflow as tf


class CNNDSSM():
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
            self.input_x1 = tf.placeholder(tf.int32, [None, max_sequence_lenth], name='input_x1')
            self.input_x2 = tf.placeholder(tf.int32, [None, max_sequence_lenth], name='input_x2')
            self.input_y = tf.placeholder(tf.int32, [None, class_num], name='input_y')

        self.l2_loss = tf.constant(0.0)

        embedding_layer1 = self.__word2vec(self.input_x1, 'embedding1')
        conv_pool_layer1 = self.__conv_pool_level(embedding_layer1, 'conv-maxpool1')
        docvec1 = self.__docvec(conv_pool_layer1, 'docvec1')

        embedding_layer2 = self.__word2vec(self.input_x2, 'embedding2')
        conv_pool_layer2 = self.__conv_pool_level(embedding_layer2, 'conv-maxpool2')
        docvec2 = self.__docvec(conv_pool_layer2, 'docvec2')

        cosine = self.__cosine(docvec1, docvec2, 'cosine')

        # Calculate mse loss
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(cosine-self.input_y), name="mse")
            self.loss = loss + l2_reg * self.l2_loss


    def __word2vec(self, input, name):
        with tf.name_scope(name):
            embedding_mat = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size]))
            word_embedded = tf.nn.embedding_lookup(embedding_mat, input)
            word_embedded = tf.expand_dims(word_embedded, -1)
        return word_embedded

    def __conv_pool_level(self, word2vec, name):
        pool_out = list()
        for filter_size in self.filter_sizes:
            with tf.name_scope('{}-{}'.format(name, filter_size)):
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

    def __docvec(self, conv_pool, name, docvec_dim=128):
        with tf.name_scope(name):
            W = tf.get_variable(
                "W",
                shape=[len(self.filter_sizes)*self.filter_num, docvec_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[docvec_dim]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            vec = tf.nn.xw_plus_b(conv_pool, W, b, name="vec")
        return vec

    def __cosine(self, vec1, vec2, name):
        with tf.name_scope(name):
            vec1_norm = tf.sqrt(tf.reduce_sum(vec1 * vec1, 1))
            vec2_norm = tf.sqrt(tf.reduce_sum(vec2 * vec2, 1))
            vec12_mul = tf.reduce_sum(vec1 * vec2, 1)
            score = tf.div(vec12_mul, vec1_norm * vec2_norm + 1e-8, name="scores")
        return score