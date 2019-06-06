from pymongo import MongoClient
import jieba.posseg
from time import time
import pickle
from math import ceil
import numpy as np
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from collections import defaultdict, Counter
from itertools import chain

#载入停用词词典
def load_stop_words(path = 'data/stopWords.txt'):
    stw = []
    with open(path, 'r') as fp:
        for line in fp:
            stw.append(line.strip())
    return stw

stop_words = load_stop_words()
stop_flags = ['b', 'c', 'e', 'g', 'h', 'k', 'l',
              'o', 's', 'u', 'w', 'x', 'y', 'z',
              'un', 'nr', 'f', 'i', 'm', 'p',
              'q', 'r', 'tg', 't']


MAX_LEN = 200

#载入案由预测过程词典
def load_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    return dic
WordDic = load_dict('data/trainSet/classifier/dictionary.dic')

#载入各类别内部词典
cls = ['9001', '9012', '9047', '9130', '9299',
       '9461', '9483', '9542', '9705', '9771']
WordDicRank = dict()
for c in cls:
    WordDicRank[c] = load_dict('data/trainSet/rank/dict/'+c+'_dictionary.dic')


#载入案由预测模型
def load_case_reason_model(model_path, cls_dict_path):
    def load_cls_weight(path):
        with open(path, 'rb') as fp:
            dic = pickle.load(fp)
        weight_dic = [[item[0], item[1]] for _, item in dic.items()]
        weight_dic.sort(key=lambda x:x[0])
        weight = np.array([item[1] for item in weight_dic])
        return weight
    cls_weight = load_cls_weight(cls_dict_path)
    def my_loss(y_true, y_pred):
        gamma = 2
        alpha = np.max(y_true * cls_weight, axis=-1)
        tmp = np.max(y_true * y_pred, axis=-1)
        return -K.mean(alpha * K.pow(1. - tmp, gamma) * K.log(K.clip(tmp, 1e-8, 1.0)))

    def my_metric(y_true, y_pred):
        predictions = K.argmax(y_pred)
        correct_predictions = K.equal(predictions, K.argmax(y_true))
        return K.mean(K.cast(correct_predictions, "float"))

    model = load_model(model_path, {'my_loss': my_loss, 'my_metric': my_metric})
    return model

CaseReasonModel = load_case_reason_model(
    model_path = 'data/trainSet/classifier/training/TextCNN/model.h5',
    cls_dict_path = 'data/trainSet/classifier/cls_5w.dic'
)


#载入案由映射词典
def load_cls_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    weight_dic = [[item[0], cls] for cls, item in dic.items()]
    weight_dic.sort(key=lambda x:x[0])
    cls_list = [item[1] for item in weight_dic]
    return cls_list

ClsDict = load_cls_dict('data/trainSet/classifier/cls_5w.dic')


CONN = MongoClient('172.19.241.248', 20000)
db = CONN['wangxiao']


#载入各类别内部重排序模型
RankModel = dict()
def load_rank_model(model_path):
    def my_loss(y_true, y_pred):
        return K.mean(K.pow(K.log(y_pred+1)-K.log(y_true+1), 2))
    model = load_model(model_path, {'my_loss': my_loss})
    return model

for c in cls:
    RankModel[c] = load_rank_model('data/trainSet/rank/'+c+'model.h5')


time_res = {
    'token': [],
    'predict': [],
    'search': [],
    'rank': [],
    'out': [],
}

def TimeCost(name, tag):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time()
            func(*args, **kwargs)
            end = time()
            t = end - start
            print('{} : {} s'.format(name, t))
            time_res[tag].append(t)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class CaseStatutesRecommend:
    def __init__(self):
        pass

    def __doc_preprocess(self, doc, word_dict):
        doc_code = list(map(lambda w: word_dict[w] if w in word_dict else 0, doc))
        X = np.array([(doc_code * ceil(MAX_LEN / len(doc_code)))[:MAX_LEN]])
        return X

    @TimeCost('----分词', 'token')
    def __token(self, input):
        token = jieba.posseg.cut(input.strip())
        token = [x.word for x in filter(lambda x: x.flag not in stop_flags and x.word not in stop_words, token)]
        return token

    @TimeCost('----案由预测', 'predict')
    def __reason_predict(self, token):
        X = self.__doc_preprocess(token, WordDic)

        y_pred = CaseReasonModel.predict(X)
        y_pred = K.argmax(y_pred)
        with tf.Session() as sess:
            y_pred = y_pred.eval()
        reason = ClsDict[y_pred]
        return reason

    @TimeCost('----类案搜索', 'search')
    def __case_search(self, token, caseReason):
        col = db['tfidf_'+caseReason]
        tmp = defaultdict(lambda: 0)
        for word in token:
            find_res = col.find_one({'word': word})
            if find_res:
                for r in find_res['doc_tfidf']:
                    tmp[r['doc']] += r['tfidf']
        tmp = [[k, v] for k, v in dict(tmp).items()]
        tmp.sort(key=lambda x: x[1], reverse=True)
        tmp = tmp[:100]
        tfidf_max = max([x[1] for x in tmp])
        tfidf_min = min([x[1] for x in tmp])
        t = tfidf_max-tfidf_min
        SimCaseCandidateSet = [[x[0], (x[1]-tfidf_min)/t] for x in tmp]
        return SimCaseCandidateSet

    @TimeCost('----相似度打分', 'rank')
    def __rank(self, token, caseReason, SimCaseCandidateSet, alpha=0.3):
        col = db['alldata_final']
        word_dict = WordDicRank[caseReason]
        model = RankModel[caseReason]

        X1 = self.__doc_preprocess(token, word_dict)

        res = []
        for case in SimCaseCandidateSet:
            id = case[0]
            sim_tfidf = case[1]

            try:
                find_res = col.find_one({'fullTextId': id})
                doc = find_res['rough_cleaned']
                refs = find_res['reference']
            except:
                continue

            X2 = self.__doc_preprocess(doc.strip().replace('。', ' ').split(' '), word_dict)
            sim_statute = model.predict([X1, X2])

            sim = alpha*sim_tfidf + (1-alpha)*sim_statute

            refs = [ref['name']+'-'+ref['levelone'] for ref in refs]
            res.append([id, sim, sim_statute, refs])

        return res

    @TimeCost('----输出', 'out')
    def __out_cases_statutes(self, rankRes, k=20, b=0.5):
        rankRes.sort(key=lambda x: x[1], reverse=True)
        case_list = [item[0] for item in rankRes[:k]]

        case_tmp = list(filter(lambda x: x[2] > b, rankRes))
        while len(case_tmp)==0:
            b -= 0.1
            case_tmp = list(filter(lambda x: x[2] > b, rankRes))

        statutes = map(lambda x: x[3], case_tmp)
        statutes = chain(*statutes)
        statutes_count = Counter(statutes)

        statutes_list = [[k, v] for k, v in statutes_count.items()]
        statutes_list.sort(key=lambda x: x[1], reverse=True)

        return case_list, statutes_list


    def recommend(self, inputs):
        for input in inputs:
            token = self.__token(input)
            reason = self.__reason_predict(token)
            SimCaseCandidateSet = self.__case_search(token, reason)
            rankRes = self.__rank(token, reason, SimCaseCandidateSet)
            case_list, statutes_list = self.__out_cases_statutes(rankRes)

            print(case_list, statutes_list)