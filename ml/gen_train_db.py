#coding:utf-8

from pymongo import MongoClient
import pandas as pd
import re
import os
import pickle


p_re = re.compile(r'.*诉\s*称\s*，\s*|.*诉\s*称\s*：\s*')
f_re = re.compile(r'.*审理\s*查明\s*，\s*|.*审理\s*查明\s*：\s*|.*本院.*事实.*?：\s*')

level = re.compile(r'(\d{4})-9000')


conn = MongoClient('192.168.68.11', 20000)

db1 = conn.lawCase
db2 = conn.wx_data


def get_all_info_data(csv_path='../data/case_info.csv'):

    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        return data

    statutes = []

    set1 = db1.paragraph
    for item in set1.find():
        statutes.append([item['fullTextId'], item['codeOfCauseOfAction']])

    statutes_df = pd.DataFrame(statutes, columns=['id', 'statute_code'])
    def statute_col_helper(x):
        try:
            return int(x)
        except:
            return None

    statutes_df['statute_code'] = statutes_df['statute_code'].apply(statute_col_helper)
    statutes_df.dropna(how='any', inplace=True)

    seg_data = []

    set2 = db1.AJsegment
    for item in set2.find():
        tmp = [item['fulltextid']]

        if item['plaintiffAlleges'] is not None and 'token' in item['plaintiffAlleges']:
            tmp.append(item['plaintiffAlleges']['token'])
            tmp.append(item['plaintiffAlleges']['flag'])
        else:
            tmp.append(None)

        if item['factFound'] is not None and 'token' in item['factFound']:
            tmp.append(item['factFound']['token'])
            tmp.append(item['factFound']['flag'])
        else:
            tmp.append(None)

        seg_data.append(tmp)

    seg_df = pd.DataFrame(seg_data, columns=['id', 'plaintiff_token', 'plaintiff_flag', 'fact_token', 'fact_flag'])

    all_info = pd.merge(seg_df, statutes_df, on='id', how='inner')


    def fact_col_helper(row):
        if pd.isnull(row['fact']):
            return None
        else:
            if f_re.search(row['fact']):
                return re.sub(f_re, '', row['fact']).strip()
            elif pd.notnull(row['plaintiff']) and p_re.search(row['plaintiff']):
                return re.sub(p_re, '', row['plaintiff']).strip()
            else:
                return None

    all_info['text'] = all_info.apply(fact_col_helper, axis=1)

    data_df = all_info.loc[:, ['text', 'statute_code']]
    data_df.dropna(how='any', inplace=True)
    data_df.to_csv(csv_path, index=0)

    return data_df


def get_code_action(code_path = '../data/statute_code.csv'):
    if os.path.exists(code_path):
        data = pd.read_csv(code_path)
        return data

    codeofca = db.codeofca

    statute_code = []

    for i in codeofca.find():
        statute_code.append([i['currentcode'], i['causeofaction'], i['tree']])

    statute_code.sort(key=lambda x: x[0])

    df = pd.DataFrame(statute_code, columns=['code', 'name', 'tree'])

    print(df)

    df.to_csv('../data/statute_code.csv', index=0)
    return df


def code_helper(x):
    tmp = level.search(x)
    if tmp:
        return tmp.group(1)
    else:
        return None


def get_code_map_dict(dict_path = '../data/code_map.dict'):
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as fp:
            dict = pickle.load(fp)
            return dict

    df = get_code_action()

    df['std'] = df['tree'].apply(code_helper)
    df.dropna(inplace=True)

    res = {}

    for index, row in df.iterrows():
        res[str(row['code'])] = row['std']

    with open(dict_path, 'wb') as fp:
        pickle.dump(res, fp)

    return res



if __name__ == '__main__':
    dic = get_code_map_dict()
    print(dic)
