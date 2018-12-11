# coding:utf-8
import pandas as pd
import random
import fasttext



def load_stop_words(path = 'data/stopWords.txt'):
    stw = []
    with open(path, 'r') as fp:
        for line in fp:
            stw.append(line.strip())
    return stw

stop_words = load_stop_words()
stop_flags = ['']



def preprocess_str(string, with_flag = True):
    res = []
    for wf in string.split(' '):
        word, flag = wf.split('-')
        if word not in stop_words:
            if with_flag and flag not in stop_flags:
                res.append(word)
            else:
                res.append(word)
    return res



def read_dataset(data_path):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    data1 = df[df['tag'] == True]
    data2 = df[df['tag'] == False]

    # Train set
    x_train = [preprocess_str(x) for x in data1['text'].tolist()]
    y_train = data1['label'].tolist()

    train_set = []
    for x, y in zip(x_train, y_train):
        train_set.append("__lable__"+str(int(y))+" , " + " ".join(x))

    # Test set
    x_test = [preprocess_str(x) for x in data2['text'].tolist()]
    y_test = data2['label'].tolist()

    test_set = []
    for x, y in zip(x_test, y_test):
        test_set.append("__lable__" + str(int(y)) + " , " + " ".join(x))

    return train_set, test_set



def writeData(sentences, fileName):
    random.shuffle(sentences)
    print("writing data to fasttext format...")
    with open(fileName, 'w', encoding='utf-8') as fp:
        for sentence in sentences:
            fp.write(sentence+"\n")
    print("done!")



if __name__=="__main__":
    data_file = r'data/data.csv'

    train_set_file = r'data/train_data/fasttext_train_data.txt'
    test_set_file = r'data/train_data/fasttext_test_data.txt'

    train_set, test_set = read_dataset(data_file)
    writeData(train_set, train_set_file)
    writeData(test_set, test_set_file)

    classifier=fasttext.supervised(train_set_file,'model/Fasttext_classifier.model',lable_prefix='__lable__')
    result = classifier.test(test_set_file)
    print("P@1:",result.precision)    #准确率
    print("R@2:",result.recall)    #召回率
    print("Number of examples:",result.nexamples)    #预测错的例子

    # #实际预测
    # lable_to_cate={1:'technology'.1:'car',3:'entertainment',4:'military',5:'sports'}
    #
    # texts=['中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟 主场 作战 中国队 率先 打破 场上 僵局 利用 角球 机会 大宝 前点 攻门 得手 中国队 领先']
    # lables=classifier.predict(texts)
    # print(lables)
    # print(lable_to_cate[int(lables[0][0])])
    #
    # #还可以得到类别+概率
    # lables=classifier.predict_proba(texts)
    # print(lables)
    #
    # #还可以得到前k个类别
    # lables=classifier.predict(texts, k=3)
    # print(lables)
    #
    # #还可以得到前k个类别+概率
    # lables=classifier.predict_proba(texts, k=3)
    # print(lables)