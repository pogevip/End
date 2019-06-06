from pymongo import MongoClient
import jieba.posseg
import pickle
from math import ceil
import numpy as np
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from collections import defaultdict, Counter
from itertools import chain
from Rank.DSSM import CosineLayer
import os

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

def load_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    return dic

#载入案由预测过程词典
WordDic = load_dict('data/trainSet/classifier/dictionary.dic')

#载入各类别内部词典
cls = ['9001', '9012', '9047', '9130', '9299',
       '9461', '9483', '9542', '9705', '9771']
WordDicRank = {c : load_dict(os.path.join('data/trainSet/rank/dict/', c+'_dictionary.dic')) for c in cls}

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

    def myacc(y_true, y_pred):
        predictions = K.argmax(y_pred)
        correct_predictions = K.equal(predictions, K.argmax(y_true))
        return K.mean(K.cast(correct_predictions, "float"))

    model = load_model(model_path, {'my_loss': my_loss, 'myacc': myacc})
    return model

CaseReasonModel = load_case_reason_model(
    model_path = 'model/CRPM.hdf5',
    cls_dict_path = 'data/trainSet/classifier/cls_5w.dic'
)
print('CRPM load finished!')

#载入法条相关性预测模型
def load_SCPM(model_path):
    def myloss(y_true, y_pred):
        return K.mean(K.pow(K.log(y_pred+1)-K.log(y_true+1), 2))

    model = load_model(model_path, {'myloss': myloss, 'CosineLayer': CosineLayer})
    return model

SCPMs = {}
for c in cls:
    SCPMs[c] = load_SCPM(os.path.join('model', 'SCPM_'+c+'.hdf5'))
    print('SCPMS ' + c + ' load finished!')
print('SCPMS load finished!')

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


class CaseStatutesRecommend:
    def __init__(self):
        pass

    def __token(self, input):
        token = jieba.posseg.cut(input.strip())
        token = [x.word for x in filter(lambda x: x.flag not in stop_flags and x.word not in stop_words, token)]
        return token

    def __doc_preprocess(self, doc, word_dict):
        doc_code = list(map(lambda w: word_dict[w] if w in word_dict else 0, doc))
        X = np.array([(doc_code * ceil(MAX_LEN / len(doc_code)))[:MAX_LEN]])
        return X

    def __reason_predict(self, token):
        X = self.__doc_preprocess(token, WordDic)
        y_pred = np.argmax(CaseReasonModel.predict(X), axis=1)[0]

        return str(ClsDict[y_pred])

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
        CandidateSet = [[x[0], (x[1]-tfidf_min)/t] for x in tmp]
        return CandidateSet

    def __rank(self, token, caseReason, CandidateSet, alpha=0.3):
        col = db['web_case_info']
        word_dict = WordDicRank[caseReason]
        model = SCPMs[caseReason]

        X = self.__doc_preprocess(token, word_dict)[0]
        refs = []
        sr = []

        for case in CandidateSet:
            id = case[0]

            try:
                find_res = col.find_one({'fulltextid': id})
                doc_code = self.__doc_preprocess(find_res['token'].strip().replace('。', ' ').split(' '), word_dict)[0]
                refs.append([r['name']+' '+r['levelone'] for r in find_res['references']])
                sr.append(model.predict([np.array(X).reshape(1,MAX_LEN), np.array(doc_code).reshape(1,MAX_LEN)])[0])
            except:
                continue
        
        print(sr)
        sr_min = min(sr)
        step = max(sr)-sr_min
        sr = [(s-sr_min)/step for s in sr]

        res = [{'id':s1[0],
                's1':s1[1],
                's2':s2,
                'sim':alpha*s1[1] + (1-alpha)*s2,
                'ref':r}
               for s1, s2, r in zip(CandidateSet, sr, refs)]
        return res


    def __out_cases_statutes(self, rankRes, k=20, b=0.7):
        rankRes.sort(key=lambda x: x['sim'], reverse=True)
        case_list = [item['id'] for item in rankRes[:k]]

        case_tmp = list(filter(lambda x: x['s2'] > b, rankRes))
        while len(case_tmp)==0:
            b -= 0.1
            case_tmp = list(filter(lambda x: x[2] > b, rankRes))

        statutes = map(lambda x: x['ref'], case_tmp)
        statutes = chain(*statutes)
        statutes_count = Counter(statutes)

        statutes_list = [[k, v] for k, v in statutes_count.items()]
        statutes_list.sort(key=lambda x: x[1], reverse=True)

        return case_list, statutes_list


    def recommend(self, inputs):
        for input in inputs:
            token = self.__token(input)
            reason = self.__reason_predict(token)
            print(reason)
            SimCaseCandidateSet = self.__case_search(token, reason)
            rankRes = self.__rank(token, reason, SimCaseCandidateSet)
            print(rankRes)
            case_list, statutes_list = self.__out_cases_statutes(rankRes)

            print(case_list, statutes_list)