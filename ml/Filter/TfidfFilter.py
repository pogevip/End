#coding:utf-8

import os
import pickle
import pandas as pd


class TfidfFilter():
    def __init__(self):
        with open('', 'rb') as fp:
            self.dictionary = pickle.load(fp)
        self.tfidf_df = pd.read_csv('')
        self.case_id_df = pd.read_csv('')


    def get_topk_case_id(self, inputs, k=100):
        word_indexs = []
        for word in inputs:
            if word in self.dictionary:
                word_indexs.append(self.dictionary[word])

        filter_df = self.tfidf_df[self.tfidf_df['word_index'].isin(word_indexs)]

        tmp = filter_df['tfidf'].groupby(filter_df['name_index']).sum()
        index = tmp.sort_values(ascending=False).head(k).index.values

        candidate_case_ids = self.case_id_df.loc[index]

        return candidate_case_ids['id'].tolist()
