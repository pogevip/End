# coding:utf-8
from TextClassifier.TextCNN_LDA import TextCNN_LDA

from TextClassifier.data_helper import load_data, load_cls_weight, LdaRandomSampleDataGenerator
import numpy as np
from keras import optimizers, callbacks
import keras.backend as K
import math
import os


EPOCH = 10

BATCH_SIZE = 256
BATCH_PROP = 0.0005
DIC_SIZE = 81562


def train(data_path,
          class_weight_path,
          out_path,
          batch_size=BATCH_SIZE,
          epoch=EPOCH,
          ):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print('load_data')
    train_data = load_data(os.path.join(data_path, 'train.pkl'))
    train_data_num = len(train_data['X'])
    train_generator = LdaRandomSampleDataGenerator(train_data, batch_size=batch_size)

    val_data = load_data(os.path.join(data_path, 'val.pkl'))
    val_data_num = len(val_data['X'])
    val_generator = LdaRandomSampleDataGenerator(val_data, batch_size=batch_size)

    cls_weight = load_cls_weight(class_weight_path)

    model = TextCNN_LDA(dict_size=DIC_SIZE)

    print(model.summary())

    def my_loss(y_true, y_pred):
        gamma = 2
        alpha = np.max(y_true * cls_weight, axis=-1)
        tmp = np.max(y_true * y_pred, axis=-1)
        return -K.mean(alpha * K.pow(1. - tmp, gamma) * K.log(K.clip(tmp, 1e-8, 1.0)))

    def myacc(y_true, y_pred):
        predictions = K.argmax(y_pred)
        correct_predictions = K.equal(predictions, K.argmax(y_true))
        return K.mean(K.cast(correct_predictions, "float"))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=my_loss, optimizer=adam, metrics=[myacc])

    checkpointer = callbacks.ModelCheckpoint(
        filepath=os.path.join(out_path, 'name_epoch{epoch:02d}_valacc{val_myacc:.3f}.hdf5'),
        monitor='val_myacc', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=0)
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

    data_path = os.path.join(dir, 'web_train_file_lda_att')
    out_path = dir + 'training/TextCNN_lda_att'
    train(data_path, out_path)