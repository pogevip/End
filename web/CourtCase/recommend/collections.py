import pymongo
from bson.objectid import ObjectId
from django.conf import settings
from collections import defaultdict



class indexCol:
    def __init__(self, cls):
        self.col = settings.CONN['wangxiao']['tfidf_'+str(cls)]

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

        return tmp


class CaseTokenCol:
    def __init__(self):
        self.col = settings.CONN['wangxiao']['alldata_final']

    def getInfo(self, id):
        item = self.col.find_one({"fullTextId" : id})
        return item


class AllInfoCol:
    def __init__(self):
        self.col = settings.CONN['lawCase']['lawcase']

    def getSummary(self, id):
        item = self.col.find_one({"_id" : ObjectId(id)})
        return item['text']

    def getAllInfo(self, id):
        item = self.col.find_one({"_id": ObjectId(id)})
        return item