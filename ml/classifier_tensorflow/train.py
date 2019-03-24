from classifier_tensorflow import TextCNN_train

train_data_path = 'data/trainSet/TextCnn/rough/train'
val_data_path = 'data/trainSet/TextCnn/rough/val'
vacab_data_path = 'data/trainSet/vocab.dic'

TextCNN_train.train(train_data_path, val_data_path, vacab_data_path)