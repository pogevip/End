#coding:utf-8

from pymongo import MongoClient
import pandas as pd
import re
import os
import jieba.posseg
import pickle


p_re = re.compile(r'.*诉称\s*，\s*|.*诉称\s*：\s*')
f_re = re.compile(r'.*审理查明\s*，\s*|.*审理查明\s*：\s*|.*本院.*?事实.*?：\s*')


conn = MongoClient('172.19.241.248', 20000)

db = conn.lawCase

to_set = conn.wangxiao.Alldata



def get_case_code(case_code_path = '../data/case_code.csv'):
    if os.path.exists(case_code_path):
        case_code = pd.read_csv(case_code_path)
    else:
        set = db.paragraph
        statutes = []
        i = 0
        for item in set.find():

            i += 1
            if i % 5000 == 0:
                print('case_code: ' + str(i))

            statutes.append([item['fullTextId'], item['codeOfCauseOfAction']])

        case_code = pd.DataFrame(statutes, columns=['id', 'statute_code'])

        print('case_code read finished!')

        def statute_col_helper(x):
            try:
                return int(x)
            except:
                return None

        case_code['statute_code'] = case_code['statute_code'].apply(statute_col_helper)
        case_code.dropna(how='any', inplace=True)

        print('case_code to csv...')
        case_code.to_csv(case_code_path, index=0)

    return case_code


def get_case_text(seg_info_path = '../data/case_text.csv'):
    if os.path.exists(seg_info_path):
        seg_df = pd.read_csv(seg_info_path)
    else:
        set = db.AJsegment
        seg_data = []
        i = 0
        for item in set.find():

            i += 1
            if i % 5000 == 0:
                print('seg_info: ' + str(i))

            tmp = [item['fulltextid']]

            if item['factFound'] is not None and 'text' in item['factFound'] and f_re.search(item['factFound']['text']):
                tmp.append(re.sub(f_re, '', item['factFound']['text']).strip())
                tmp.append(True)
            elif item['plaintiffAlleges'] is not None and 'text' in item['plaintiffAlleges'] and p_re.search(
                    item['plaintiffAlleges']['text']):
                tmp.append(re.sub(p_re, '', item['plaintiffAlleges']['text']).strip())
                tmp.append(False)
            else:
                continue
            seg_data.append(tmp)

        seg_df = pd.DataFrame(seg_data, columns=['id', 'text', 'is_fact'])
        print('case text to csv...')
        seg_df.to_csv(seg_info_path, index=0)

    return seg_df


def get_all_info_tmp_data(statutes_df, case_df, all_info_tmp_path = '../data/all_info_tmp.csv'):

    if os.path.exists(all_info_tmp_path):
        all_info_tmp = pd.read_csv(all_info_tmp_path)
    else:
        print('text_info and case_code merging...')
        all_info_tmp = pd.merge(case_df, statutes_df, on='id', how='inner')

        with open('../data/code_map.dict', 'rb') as fp:
            code_map = pickle.load(fp)

        all_info_tmp['cls'] = all_info_tmp['statute_code'].apply(
            lambda x: code_map[str(int(x))] if str(int(x)) in code_map else None)
        all_info_tmp.dropna(how='any', inplace=True)

        print('all_info_tmp to csv...')
        all_info_tmp.to_csv(all_info_tmp_path, index=0)

    return all_info_tmp


def gen_all_data_to_db(all_info_tmp):

    def segment(sentence):
        sentence_seged = jieba.posseg.cut(sentence.strip())
        sentence_seged = filter(lambda x: x.word != ' ' and x.word != '/', sentence_seged)
        sentence_seged = map(lambda x: "{}/{}".format(x.word, x.flag), sentence_seged)
        return ' '.join(sentence_seged)

    all_info_tmp.dropna(how='any', inplace=True)

    print('start segment...')

    buffer = []
    for index, row in all_info_tmp.iterrows():
        if index % 2000 == 0:
            print(index)
        buffer.append({
            'id': row['id'],
            'text': row['text'],
            'token': segment(row['text']),
            'is_fact': row['is_fact'],
            'statute_code': str(int(row['statute_code'])),
            'cls': row['cls']
        })
        if len(buffer) > 20000:
            to_set.insert_many(buffer)
            buffer.clear()
    if len(buffer) > 0:
        to_set.insert_many(buffer)
        buffer.clear()

    print('save db finished !!!')


def gen_all_info_df(all_info_path = '../data/all_info.csv'):
    if os.path.exists(all_info_path):
        all_info = pd.read_csv(all_info_path)
    else:
        all_info = []
        i = 0
        for item in to_set.find():

            i += 1
            if i % 5000 == 0:
                print('case_code: ' + str(i))

            all_info.append([item['id'], item['token'], item['is_fact'], item['statute_code'], item['cls']])

        all_info = pd.DataFrame(all_info, columns=['id', 'token', 'is_fact', 'statute_code', 'cls'])

        print('all_info to csv...')
        all_info.to_csv(all_info_path, index=0)

    return all_info


def gen_all_info_df2(all_info_tmp, all_info_path = '../data/all_info.csv'):

    def segment(sentence):
        sentence_seged = jieba.posseg.cut(sentence.strip())
        sentence_seged = filter(lambda x: x.word != ' ' and x.word != '/', sentence_seged)
        sentence_seged = map(lambda x: "{}/{}".format(x.word, x.flag), sentence_seged)
        return ' '.join(sentence_seged)

    if os.path.exists(all_info_path):
        all_info = pd.read_csv(all_info_path)
    else:
        all_info_tmp['token'] = all_info_tmp['text'].apply(segment)

        all_info = all_info_tmp.loc[:,['id', 'token', 'is_fact', 'statute_code', 'cls']]

        print('all_info to csv...')
        all_info.to_csv(all_info_path, index=0)

    return all_info


if __name__ == '__main__':
    # statutes_df = get_case_code()
    # case_df = get_case_text()
    statutes_df, case_df = None, None
    all_info_tmp = get_all_info_tmp_data(statutes_df, case_df)
    all_info = gen_all_data_to_db(all_info_tmp)
    gen_all_info_df()
    # gen_all_info_df2(all_info_tmp)

    print('finished !!!')