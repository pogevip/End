import fasttext


if __name__ == '__main__':
    train_file_path = '/Users/wangxiao/Desktop/fasttext_data/trainData.txt'
    test_file_path = '/Users/wangxiao/Desktop/fasttext_data/testData.txt'

    model_path = '/Users/wangxiao/Desktop/fasttext_data/model'
    label_prefix = '__lable__'

    print('training...')
    model = fasttext.supervised(train_file_path, model_path, label_prefix=label_prefix)

    print('testing...')
    model = fasttext.load_model(model_path + '.bin')
    result = model.test(test_file_path)
    print("P@1:", result.precision)  # 准确率
    print("R@2:", result.recall)  # 召回率
    print("Number of examples:", result.nexamples)


###############################
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