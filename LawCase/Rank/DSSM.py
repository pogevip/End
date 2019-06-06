#coding:utf-8

from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from Rank.Semantic import CNNSem, LSTMSem, BiLSTMAttSem, BiLSTMSem
# from Semantic import CNNSem, LSTMSem, BiLSTMAttSem, BiLSTMSem

import tensorflow as tf
from keras.layers.core import Layer


MAX_LEN = 200


class CosineLayer(Layer):
    def __init__(self,  **kwargs):
        super(CosineLayer, self).__init__(**kwargs)  ## 继承Layer中的初始化参数

    def build(self, input_shape):
        super(CosineLayer, self).build(input_shape)

    def call(self, input):
        query = input[0]
        doc = input[1]
        pooled_len_1 = tf.sqrt(tf.reduce_sum(query * query, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(doc * doc, 1))
        pooled_mul_12 = tf.reduce_sum(query * doc, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
        return score

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)



def DSSM(dict_size,  semantic, squence_len=200, embed_size=256):
    query_seq = Input(shape=[squence_len], name='q_seq')
    doc_seq = Input(shape=[squence_len], name='d_seq')

    # Embeddings layers
    query_emb = Embedding(input_dim=dict_size+1, output_dim=embed_size, input_length=squence_len)(query_seq)
    doc_emb = Embedding(input_dim=dict_size+1, output_dim=embed_size, input_length=squence_len)(doc_seq)
    #input_dim:词典大小+1
    #out_dim:输出词向量大小
    #input_lenth:句子最大长度

    if semantic == 'cnn':
        query_sem = CNNSem(query_emb)
        doc_sem = CNNSem(doc_emb)
    elif semantic == 'lstm':
        query_sem = LSTMSem(query_emb)
        doc_sem = LSTMSem(doc_emb)
    elif semantic == 'bilstm':
        query_sem = BiLSTMSem(query_emb)
        doc_sem = BiLSTMSem(doc_emb)
    elif semantic == 'bilstmAtt':
        query_sem = BiLSTMAttSem(query_emb)
        doc_sem = BiLSTMAttSem(doc_emb)
    else:
        raise ValueError('param "seamntic" must in ["cnn", "lstm", "bilstm", "bilstmAtt"]')

    cosine_sim = CosineLayer()([query_sem, doc_sem])

    model = Model([query_seq, doc_seq], cosine_sim)
    return model


if __name__ == '__main__':
    model = DSSM(8251, 'cnn')
    print(model.summary())