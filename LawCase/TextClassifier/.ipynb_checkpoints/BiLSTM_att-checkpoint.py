#coding:utf-8

from keras.layers import *
from keras.models import *


MAX_LEN = 200


def attention_3d_block(inputs, time_steps):
    a = Permute((2, 1), name='att_permute')(inputs)
    a = Dense(time_steps, activation='softmax', name='att_softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output = multiply([inputs, a_probs], name='attention_mul')
    return output


def BiLSTMAtt(dict_size, squence_len=200, embed_size=256):
    # Inputs
    comment_seq = Input(shape=[squence_len], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(input_dim=dict_size+1, output_dim=embed_size, input_length=squence_len)(comment_seq)
    #input_dim:词典大小+1
    #out_dim:输出词向量大小
    #input_lenth:句子最大长度

    bilstm = Bidirectional(LSTM(units=64, return_sequences=True))(emb_comment)

    att = attention_3d_block(bilstm, squence_len)

    out = Dropout(0.3)(att)

    flat = Flatten()(out)

    output = Dense(256, activation='relu')(flat)

    output = Dense(units=10, activation='softmax')(output)

    model = Model([comment_seq], output)
    return model

if __name__ == '__main__':
    model = BiLSTMAtt(8251)
    print(model.summary())