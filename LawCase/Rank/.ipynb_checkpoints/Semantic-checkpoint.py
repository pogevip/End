#coding:utf-8

from keras.layers import *

MAX_LEN = 200

def CNNSem(emb, out_dim=128, seq_len = MAX_LEN, filter_sizes=[3,4,5], filter_num = 256):
    convs = []
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=filter_num, kernel_size=fsz, activation='relu')(emb)
        l_pool = MaxPooling1D(seq_len - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.3)(merge)
    sem = Dense(out_dim, activation='relu')(out)
    return sem


def LSTMSem(emb, out_dim=128):
    bilstm = LSTM(units=512, return_sequences=False)(emb)
    dropout = Dropout(0.3)(bilstm)
    sem = Dense(units=out_dim, activation='relu')(dropout)
    return sem


def BiLSTMSem(emb, out_dim=128):
    bilstm = Bidirectional(LSTM(units=256, return_sequences=False))(emb)
    dropout = Dropout(0.3)(bilstm)
    sem = Dense(units=out_dim, activation='relu')(dropout)
    return sem


def attention_3d_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output = multiply([inputs, a_probs])
    return output

def BiLSTMAttSem(emb, out_dim=128, seq_len = MAX_LEN):
    bilstm = Bidirectional(LSTM(units=128, return_sequences=True))(emb)
    att = attention_3d_block(bilstm, seq_len)
    out = Dropout(0.3)(att)
    flat = Flatten()(out)
    sem = Dense(out_dim, activation='relu')(flat)
    return sem


if __name__ == '__main__':
    pass