#coding:utf-8

from keras.layers import Dense, LSTM, Bidirectional, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model


MAX_LEN = 200

def BiLSTM(dict_size, squence_len=200, embed_size=256):
    # Inputs
    comment_seq = Input(shape=[squence_len], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(input_dim=dict_size+1, output_dim=embed_size, input_length=squence_len)(comment_seq)
    #input_dim:词典大小+1
    #out_dim:输出词向量大小
    #input_lenth:句子最大长度

    bilstm = Bidirectional(LSTM(units=128, return_sequences=False))(emb_comment)

    dropout = Dropout(0.3)(bilstm)

    output = Dense(units=10, activation='softmax')(dropout)

    model = Model([comment_seq], output)
    return model


if __name__ == '__main__':
    model = BiLSTM(8251)
    print(model.summary())