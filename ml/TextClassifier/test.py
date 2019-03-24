# coding:utf-8

import os, math
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from TextClassifier.data_helper import load_data, MAX_LEN, load_cls_dict, load_cls_weight
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

params = {
    'axes.labelsize': '20',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'lines.linewidth': '1',
    'figure.figsize': '15, 15'  # set figure size
}
pylab.rcParams.update(params)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", size=20,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def test(data_path, model_path, cls_dic_path):
    print('load data')
    test_data = load_data(os.path.join(data_path, 'test.pkl'))
    X = test_data['X']
    X = np.array(list(map(lambda doc: (doc * math.ceil(MAX_LEN / len(doc)))[:MAX_LEN], X)))

    y_true = test_data['Y']

    print('load model')

    cls_weight = load_cls_weight(cls_dic_path)

    def my_loss(y_true, y_pred):
        gamma = 2
        alpha = np.max(y_true * cls_weight, axis=-1)
        tmp = np.max(y_true * y_pred, axis=-1)
        return -K.mean(alpha * K.pow(1. - tmp, gamma) * K.log(K.clip(tmp, 1e-8, 1.0)))

    def my_metric(y_true, y_pred):
        predictions = K.argmax(y_pred)
        correct_predictions = K.equal(predictions, K.argmax(y_true))
        return K.mean(K.cast(correct_predictions, "float"))

    model = load_model(os.path.join(model_path, 'model.h5'), {'my_loss': my_loss, 'my_metric': my_metric})

    print('predict...')
    y_pred = model.predict(X, batch_size=500)
    y_pred = K.argmax(y_pred)
    with tf.Session() as sess:
        y_pred = y_pred.eval()
    print(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cls = load_cls_dict(cls_dic_path)

    return cm, cls

def show(cm, cls, name, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    np.save(os.path.join(out_path, name + "_cm.npy"), cm)

    np.set_printoptions(precision=2)
    plot_confusion_matrix(cm, classes=cls)
    plt.savefig(os.path.join(out_path, name + ".png"))

    # Plot normalized confusion matrix
    plot_confusion_matrix(cm, classes=cls, normalize=True)
    plt.savefig(os.path.join(out_path, name + "_norm.png"))

    plt.show()


if __name__ == '__main__':
    data_path = 'data/trainSet/classifier/train_file_5w_1'
    model_path = 'data/trainSet/classifier/training/TextCNN'
    cls_dic_path = 'data/trainSet/classifier/cls_5w.dic'
    out_path = 'data/trainSet/classifier/test'

    cm, cls = test(data_path, model_path, cls_dic_path)
    show(cm, cls, 'TextCNN', out_path)