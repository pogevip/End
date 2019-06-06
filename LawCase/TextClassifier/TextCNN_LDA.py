#coding:utf-8

from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Multiply
from keras.layers.embeddings import Embedding
from keras.models import Model

MAX_LEN = 200
BATCH_SIZE = 256

def TextCNN_LDA(dict_size, squence_len=200, lda_topic_num=40, embed_size=256):
    # Inputs
    comment_seq = Input(shape=[squence_len], name='x_seq')
    lda = Input(shape=[lda_topic_num], name='x_lda')

    # Embeddings layers
    emb_comment = Embedding(input_dim=dict_size+1, output_dim=embed_size, input_length=squence_len)(comment_seq)
    #input_dim:词典大小+1
    #out_dim:输出词向量大小
    #input_lenth:句子最大长度

    # conv layers
    convs = []
    filter_sizes = [3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=256, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(squence_len - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.3)(merge)
    output = Dense(64, activation='relu')(out)

    output = Dense(units=10, activation='relu')(output)

    lda_att = Dense(units=10, activation='softmax')(lda)

    output = Multiply()([output, lda_att])

    output = Dense(units=10, activation='softmax')(output)

    model = Model([comment_seq, lda], output)
    return model