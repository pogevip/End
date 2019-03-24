import pymongo
from bson.objectid import ObjectId
from django.conf import settings
from collections import defaultdict



class indexTable:
    def __init__(self, cls):
        self.col = settings.DB_CON['wangxiao']['tfidf_'+str(cls)]

    def find(self, word_list):
        tmp = defaultdict(lambda: 0)
        for word in word_list:
            find_res = self.col.find_one({'word': word})
            if find_res:
                for r in find_res['doc_tfidf']:
                    tmp[r['doc']] += r['tfidf']

        tmp = [(k, v) for k, v in dict(tmp).items()]
        tmp.sort(key=lambda x: x[1], reverse=True)
        tmp = tmp[:100]

        return [i[0] for i in tmp]


class CaseToken:
    def __init__(self):
        self.col = settings.DB_CON['wangxiao']['alldata_final']

    def getToken(self, id_list):
        res = []
        for id in id_list:
            item = self.col.find_one({"fullTextId" : id})
            res.append([id, item['token_cleaned']])
        return res


class AllInfo:
    def __init__(self):
        self.col = settings.DB_CON['lawCase']['lawcase']

    def getInfo(self, id):
        item = self.col.find_one({"_id" : ObjectId(id)})
        return item['text']