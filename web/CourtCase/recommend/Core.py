import jieba.posseg
import numpy as np
from math import ceil
import keras.backend as K
import tensorflow as tf

from django.conf import Settings
from .collections import indexCol, CaseTokenCol
from itertools import chain
from collections import Counter


MAX_LEN = 200



class CaseStatutesRecommend:
    def __init__(self, input):
        self.input = input

    def __doc_preprocess(self, doc, word_dict):
        doc_code = list(map(lambda w: word_dict[w] if w in word_dict else 0, doc))
        X = np.array([(doc_code * ceil(MAX_LEN / len(doc_code)))[:MAX_LEN]])
        return X


    def __token(self, input):
        token = jieba.posseg.cut(input.strip())
        token = [x.word for x in filter(lambda x: x.flag not in Settings.STFS and x.word not in Settings.STWS, token)]
        return token


    def __reason_predict(self, token):
        X = self.__doc_preprocess(token, Settings.WordDicCls)

        y_pred = Settings.CaseReasonModel.predict(X)
        y_pred = K.argmax(y_pred)
        with tf.Session() as sess:
            y_pred = y_pred.eval()
        reason = Settings.ClsDict[y_pred]
        return reason


    def __case_search(self, token, caseReason):
        col = indexCol(caseReason)
        tmp = col.find(token)
        tfidf_max = max([x[1] for x in tmp])
        tfidf_min = min([x[1] for x in tmp])
        t = tfidf_max-tfidf_min
        SimCaseCandidateSet = [[x[0], (x[1]-tfidf_min)/t] for x in tmp]
        return SimCaseCandidateSet


    def __rank(self, token, SimCaseCandidateSet, alpha=0.3):
        col = CaseTokenCol()
        word_dict = Settings.WordDicRank
        model = Settings.RankModel

        X1 = self.__doc_preprocess(token, word_dict)

        res = []
        for case in SimCaseCandidateSet:
            id = case[0]
            sim_tfidf = case[1]

            try:
                find_res = col.getInfo(id)
                doc = find_res['rigour_cleaned']
                refs = find_res['reference']
            except:
                continue

            X2 = self.__doc_preprocess(doc.strip().replace('。', ' ').split(' '), word_dict)
            sim_statute = model.predict([X1, X2])

            sim = alpha*sim_tfidf + (1-alpha)*sim_statute

            refs = [ref['name']+'-'+ref['levelone'] for ref in refs]
            res.append([id, sim, sim_statute, refs])

        return res


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


    def recommend(self):
        token = self.__token(self.input)
        reason = self.__reason_predict(token)
        SimCaseCandidateSet = self.__case_search(token, reason)
        rankRes = self.__rank(token, SimCaseCandidateSet)
        case_list, statutes_list = self.__out_cases_statutes(rankRes)

        return case_list, statutes_list


if __name__ == '__main__':
    pass

