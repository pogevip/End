# coding:utf-8
from TextClassifier.TextCNN import TextCNN
from TextClassifier.BiLSTM import BiLSTM
from TextClassifier.BiLSTM_att import BiLSTMAtt

from TextClassifier.data_helper import load_data, load_cls_weight, StratifiedSampleDataGenerator, RandomSampleDataGenerator
import numpy as np
from keras import optimizers, callbacks
import keras.backend as K
import math
import os

MAX_LEN = 200
EPOCH = 10
EMBEDDING_SIZE = 256
DIC_SIZE = 81562

BATCH_SIZE = 256
BATCH_PROP = 0.0005


def train(data_path,
          class_weight_path,
          out_path,
          model_option,
          batch_prop = None,
          batch_size=BATCH_SIZE,
          epoch=EPOCH,
          embed_size=EMBEDDING_SIZE
          ):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print('load_data')

    if batch_prop:
        train_data = load_data(os.path.join(data_path, 'train.pkl'))
        train_data_num = 0
        for _,v in train_data.items():
            train_data_num+=len(v['X'])
        batch_size = train_data_num * batch_prop
        train_generator = StratifiedSampleDataGenerator(train_data, batch_prop=batch_prop)
    else:
        train_data = load_data(os.path.join(data_path, 'train.pkl'))
        train_data_num = len(train_data['X'])
        train_generator = RandomSampleDataGenerator(train_data, batch_size=batch_size)

    val_data = load_data(os.path.join(data_path, 'val.pkl'))
    val_data_num = len(val_data['X'])
    val_generator = RandomSampleDataGenerator(val_data, batch_size=batch_size)

    cls_weight = load_cls_weight(class_weight_path)
    
    if model_option == 'textcnn':
        model = TextCNN(dict_size=DIC_SIZE, squence_len=MAX_LEN, embed_size=embed_size)
    elif model_option == 'bilstm':
        model = BiLSTM(dict_size=DIC_SIZE, squence_len=MAX_LEN, embed_size=embed_size)
    else:
        model = BiLSTMAtt(dict_size=DIC_SIZE, squence_len=MAX_LEN, embed_size=embed_size)

    print(model.summary())

    def my_loss(y_true, y_pred):
        gamma = 2
        alpha = np.max(y_true * cls_weight, axis=-1)
        tmp = np.max(y_true * y_pred, axis=-1)
#         return -K.mean(K.pow(1. - tmp, gamma) * K.log(K.clip(tmp, 1e-8, 1.0)))
        return -K.mean(alpha * K.pow(1. - tmp, gamma) * K.log(K.clip(tmp, 1e-8, 1.0)))

    def myacc(y_true, y_pred):
        predictions = K.argmax(y_pred)
        correct_predictions = K.equal(predictions, K.argmax(y_true))
        return K.mean(K.cast(correct_predictions, "float"))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=my_loss, optimizer=adam, metrics=[myacc])

    checkpointer = callbacks.ModelCheckpoint(
        filepath=os.path.join(out_path, 'name_epoch{epoch:02d}_valacc{val_myacc:.2f}.hdf5'),
        monitor='val_myacc', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    tensorboard = callbacks.TensorBoard(log_dir=os.path.join(out_path, 'TensorBoardLogs'), histogram_freq=0,
                                        write_graph=True, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    csvlog = callbacks.CSVLogger(os.path.join(out_path, 'log.csv'))

    print('start_train...')
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=math.ceil(train_data_num / batch_size),
                        validation_data=val_generator,
                        validation_steps=math.ceil(val_data_num / batch_size),
                        epochs=epoch,
                        callbacks=[checkpointer, tensorboard, csvlog])

    # save model and weights
    model.save(os.path.join(out_path, 'model.h5'))


if __name__ == '__main__':
    dir = '../data/trainSet/classifier/'

    # #一起存储
    # data_path = os.path.join(dir, 'train_file_5w')
    # 按类区分存储
    data_path = os.path.join(dir, 'train_file_5w_1')

    class_weight_path = os.path.join(dir, 'cls_5w.dic')

    out_path = dir + 'training/BiLSTM'
    train(data_path,
          class_weight_path,
          out_path,
          batch_prop=0.0005, )
