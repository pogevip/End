# coding:utf-8
import pandas as pd
import random
# import fasttext


def read_dataset(data_path):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)

    df = df[~df['token'].isna()]

    train_data = df[df['train_val_test'] != 3]
    for cls, group in train_data.groupby('train_val_test'):
        print(cls, len(group))
    test_data = df[df['train_val_test'] == 3]
    for cls, group in test_data.groupby('train_val_test'):
        print(cls, len(group))

    # Train set
    x_train = [x.replace('。', ' ') for x in train_data['token'].tolist()]
    y_train = train_data['cls'].tolist()

    train = []
    for x, y in zip(x_train, y_train):
        train.append("__lable__" + str(int(y)) + " , " + x)

    # Test set
    x_test = [x.replace('。', ' ') for x in test_data['token'].tolist()]
    y_test = test_data['cls'].tolist()

    test = []
    for x, y in zip(x_test, y_test):
        test.append("__lable__" + str(int(y)) + " , " + x)

    return train, test


def writeData(sentences, fileName):
    random.shuffle(sentences)
    print("writing data to fasttext format...")
    with open(fileName, 'w', encoding='utf-8') as fp:
        for sentence in sentences:
            fp.write(sentence+"\n")
    print("done!")



if __name__=="__main__":
    data_file = r'../data/trainSet/classifier/train_info_5w.csv'

    train_set_file = r'../data/trainSet/classifier/fastText/trainData.txt'
    test_set_file = r'../data/trainSet/classifier/fastText/testData.txt'

    train_set, test_set = read_dataset(data_file)
    writeData(train_set, train_set_file)
    writeData(test_set, test_set_file)

    # classifier=fasttext.supervised(train_set_file,'model/Fasttext_classifier.model',lable_prefix='__lable__')
    # result = classifier.test(test_set_file)
    # print("P@1:",result.precision)    #准确率
    # print("R@2:",result.recall)    #召回率
    # print("Number of examples:",result.nexamples)    #预测错的例子


    ################################3
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