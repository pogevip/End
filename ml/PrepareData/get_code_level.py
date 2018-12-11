#coding:utf-8

from pymongo import MongoClient
import pandas as pd
import re
import os
import pickle


level = re.compile(r'(\d{4})-9000')


conn = MongoClient('192.168.68.11', 20000)

db = conn.lawCase


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