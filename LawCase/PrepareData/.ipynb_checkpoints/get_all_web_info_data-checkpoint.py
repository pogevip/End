from pymongo import MongoClient, ASCENDING
conn = MongoClient('172.19.241.248', 20000)

src_col = conn['lawCase']['paragraph']
ref_col = conn['lawCase']['lawreference']
to_col = conn['wangxiao']['web_case_info']
to_col.create_index([('fulltextid', ASCENDING)])

import pandas as pd
import os


def gen_all_case_info_col(src_path = 'data/trainSet/rank/web_each_cls_data/', key_words_path = 'data/trainSet/'):
    CLS = ['9001', '9012', '9047', '9130', '9299',
           '9461', '9483', '9542', '9705', '9771']
    for c in CLS:
        print(c)
        key_words_df = pd.read_csv(os.path.join(key_words_path,'keywords'+c+'.csv'))
        
        data = pd.read_csv(os.path.join(src_path, c+'.csv'))
        print(len(data))
        id_list = data['id'].tolist()
        print(len(id_list))
        print('start_search')
        buffer = []
        for id in id_list:
            item = src_col.find_one({'fullTextId':id})
            ref_item = ref_col.find_one({'fullTextId':id})
            if not item or not ref_item:
                continue

            r = {}
            r['fulltextid'] = id
            try:
                r['title'] = item['title']
            except:
                pass
            try:
                defendantArgued = item['defendantArgued']['text']
                r['defendantArgued'] = defendantArgued
            except:
                pass
            try:
                r['head'] = item['head']['text']
            except:
                pass
            try:
                litigationRecord = item['litigationRecord']['text']
                r['litigationRecord'] = litigationRecord
            except:
                pass
            try:
                plaintiffAlleges = item['plaintiffAlleges']['text']
                r['plaintiffAlleges'] = plaintiffAlleges
            except:
                pass
            try:
                causeOfAction = item['causeOfAction']
                r['causeOfAction'] = causeOfAction
            except:
                pass
            try:
                factFound = item['factFound']['text']
                r['factFound'] = factFound
            except:
                pass
            try:
                decision = item['analysisProcess']['text']+item['caseDecision']['text']
                r['decision'] = decision
            except:
                pass
            try:
                refs = ref_item['references']
                r['references'] = refs
            except:
                pass
            try:
                key_words = key_words_df[key_words_df['docId']==id]['keywords'].tolist()[0]
                r['keywords'] = key_words
            except:
                pass
            
            buffer.append(r)

            if len(buffer) >= 2000:
                to_col.insert_many(buffer)
                buffer.clear()

    if len(buffer) > 0:
        to_col.insert_many(buffer)
        buffer.clear()
    

if __name__ == '__main__':
    gen_all_case_info_col(src_path = '../data/trainSet/rank/web_each_cls_data/',
                          key_words_path = '../data/trainSet/')
    print('Finished!')