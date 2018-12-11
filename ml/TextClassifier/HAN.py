#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


def lenth(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


class HAN():
    def __init__(self, vocab_size, num_classes, embedding_size=200, hidden_size=50):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_lenth')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            # x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            # y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

        word_embedded_level = self.__word2vec()
        sent_vec_level = self.__sent2vec(word_embedded_level)
        out = self.__classifier(sent_vec_level)
        # doc_vec_level = self.__doc2vec(sent_vec_level)
        # out = self.__classifier(doc_vec_level)


        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.input_y,
                logits=out,
                name='loss'
            ))

        with tf.name_scope('accuracy'):
            predict = tf.argmax(out, axis=1, name='predict')
            label = tf.argmax(self.input_y, axis=1, name='label')
            self.acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))


    def __word2vec(self):
        with tf.name_scope('embedding'):
            embedding_mat = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size]))
            #shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        return word_embedded


    def __sent2vec(self, word2vec):
        with tf.name_scope('sent_vec'):
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            #shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(word2vec, [-1, self.max_sentence_length, self.embedding_size])
            #shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoder = self.__BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            #shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec = self.__AttentionLayer(word_encoder, name='word_atten')
        return sent_vec


    def __doc2vec(self, sent2vec):
        #原理与sent2vec一样，根据文档中所有句子的向量构成一个文档向量
        with tf.name_scope('doc2vec'):
            sent_vec = tf.reshape(sent2vec, [-1, self.max_sentence_num, self.hidden_size*2])
            #shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoder = self.__BidirectionalGRUEncoder(sent_vec, name='doc_encoder')
            #shape为[batch_szie, hidden_szie*2]
            doc_vec = self.__AttentionLayer(doc_encoder, name='sent_atten')
        return doc_vec


    def __classifier(self, doc2vec):
        #最终的输出层，是一个全连接层
        with tf.name_scope('classifier'):
            out = layers.fully_connected(inputs=doc2vec, num_outputs=self.num_classes, activation_fn=None, regularizer="L2")
        return out


    def __BidirectionalGRUEncoder(self, inputs, name):
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            ((fw_out, bw_out), (_,_)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                        cell_bw=GRU_cell_bw,
                                                                        inputs=inputs,
                                                                        sequence_length=lenth(inputs),
                                                                        dtype=tf.float32)
            outputs = tf.concat((fw_out, bw_out), axis=2)
        return outputs


    def __AttentionLayer(self, inputs, name):
        #inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_size
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size*2]), name='u_context')
            #使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size*2, activation_fn=tf.nn.tanh)
            #shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_out = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_out