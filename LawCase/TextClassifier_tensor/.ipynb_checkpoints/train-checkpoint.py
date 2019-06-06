from TextClassifier import TextCNN_train

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_data_path = 'data/trainSet/TextCnn/rigour/train'
val_data_path = 'data/trainSet/TextCnn/rigour/val'
vacab_data_path = 'data/trainSet/vocab.dic'

TextCNN_train.train(train_data_path, val_data_path, vacab_data_path)