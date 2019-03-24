# coding:utf-8
from Rank.DSSM import DSSM

from Rank.data_helper import load_data, DataGenerator
from keras import optimizers, callbacks
import keras.backend as K
import math
import os

MAX_LEN = 200
EPOCH = 10
EMBEDDING_SIZE = 128

BATCH_SIZE = 256


def train(data_dir,
          cls,
          model_op,
          batch_size=BATCH_SIZE,
          epoch=EPOCH,
          embed_size=EMBEDDING_SIZE):

    out_path = os.path.join(data_dir, 'training', cls, model_op)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print('load_data')

    train_data_path = os.path.join(data_dir, 'train_file', cls, 'train.pkl')
    train_data = load_data(train_data_path)
    train_data_num = len(train_data['X1'])
    train_generator = DataGenerator(train_data, batch_size=batch_size)

    val_data_path = os.path.join(data_dir, 'train_file', cls, 'val.pkl')
    val_data = load_data(val_data_path)
    val_data_num = len(val_data['X1'])
    val_generator = DataGenerator(val_data, batch_size=batch_size)

    dict_path = os.path.join(data_dir, 'dict', cls+'_dictionary.dic')
    word_dict = load_data(dict_path)
    dic_size = len(word_dict)

    model = DSSM(dict_size=dic_size, semantic=model_op, squence_len=MAX_LEN, embed_size=embed_size)

    print(model.summary())

    def my_loss(y_true, y_pred):
        return K.mean(K.pow(K.log(y_pred+1)-K.log(y_true+1), 2))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=my_loss, optimizer=adam)

    checkpointer = callbacks.ModelCheckpoint(
        filepath=os.path.join(out_path, 'name_epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5'),
        monitor='val_acc', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=50)
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
    dir = '../data/trainSet/rank/'
    model_ops = ['cnn', 'lstm', 'bilstm', 'bilstmAtt']
    train(data_dir = dir, cls='9771', model_op=model_ops[0])
