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

src_db = 'lawCase'
to_db = 'wangxiao'

case_code_col = 'paragraph'
aj_seg_col = 'AJsegment'

all_info_col = 'allData'


def load_stop_words(path = '../data/stopWords.txt'):
    stw = []
    with open(path, 'r') as fp:
        for line in fp:
            stw.append(line.strip())
    return stw

stop_words = load_stop_words()
stop_flags1 = ['b', 'c', 'e', 'g', 'h', 'k', 'l', 'o', 's', 'u', 'w', 'x', 'y', 'z', 'un']
stop_flags2 = ['f', 'i', 'm', 'p', 'q', 'r', 'tg', 't']


class AllInfo():
    def __init__(self,):
        self.conn = conn
        self.src_db = src_db
        self.to_db = to_db

        self.case_code_col = case_code_col
        self.aj_seg_col = aj_seg_col
        self.all_info_col = all_info_col

        #案由代码分类映射关系
        self.code_map_path = '../data/code_map.dict'


    def get_case_code(self, case_code_path = '../data/case_code.csv'):
        if os.path.exists(case_code_path):
            case_code = pd.read_csv(case_code_path)
        else:
            col = self.conn[self.src_db][self.case_code_col]
            statutes = []
            i = 0
            for item in col.find():

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


    def get_case_text(self, seg_info_path = '../data/case_text.csv'):
        if os.path.exists(seg_info_path):
            seg_df = pd.read_csv(seg_info_path)
        else:
            col = self.conn[self.src_db][self.aj_seg_col]
            seg_data = []
            i = 0
            for item in col.find():

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


    def get_all_info_tmp_data(self, all_info_tmp_path = '../data/all_info_tmp.csv'):

        if os.path.exists(all_info_tmp_path):
            print('load all_info_tmp...')
            all_info_tmp = pd.read_csv(all_info_tmp_path)
        else:
            statutes_df = self.get_case_code()
            case_df = self.get_case_text()
            print('text_info and case_code merging...')
            all_info_tmp = pd.merge(case_df, statutes_df, on='id', how='inner')

            with open(self.code_map_path, 'rb') as fp:
                code_map = pickle.load(fp)

            all_info_tmp['cls'] = all_info_tmp['statute_code'].apply(
                lambda x: code_map[str(int(x))] if str(int(x)) in code_map else None)
            all_info_tmp.dropna(how='any', inplace=True)

            print('all_info_tmp to csv...')
            all_info_tmp.to_csv(all_info_tmp_path, index=0)

        return all_info_tmp


    def gen_all_data_to_db(self):

        all_info_tmp = self.get_all_info_tmp_data()

        def segment(sentence):
            sentence_seged = jieba.posseg.cut(sentence.strip())
            sentence_seged = filter(lambda x: x.word != ' ' and x.word != '/', sentence_seged)
            sentence_seged = map(lambda x: "{}/{}".format(x.word, x.flag), sentence_seged)
            return ' '.join(sentence_seged)

        def clean_sen(doc, stop_words=stop_words, stop_flags_1=stop_flags1, stop_flags_2=stop_flags2):
            rough_res = []
            rigour_res = []
            for sent in doc.split('。/x'):
                sent = sent.strip()
                ro_r = []
                ri_r = []
                for x in sent.split(' '):
                    try:
                        word, flag = x.split('/')
                        if word not in stop_words and flag not in stop_flags1:
                            ro_r.append(word)
                            if flag not in stop_flags2:
                                ri_r.append(word)
                    except:
                        continue
                if len(ro_r)>0:
                    rough_res.append(' '.join(ro_r))
                if len(ri_r)>0:
                    rigour_res.append(' '.join(ri_r))
            return '。'.join(rough_res), '。'.join(rigour_res)

        def com_len(doc):
            sents = doc.split('。')
            sent_num = len(sents)
            each_sent_len = list(map(lambda x: len(x.split(' ')), sents))
            max_sentlen = max(each_sent_len)
            doc_len = sum(each_sent_len)
            return sent_num, max_sentlen, doc_len

        print('start segment...')

        col = self.conn[self.to_db][self.all_info_col]

        buffer = []
        for index, row in all_info_tmp.iterrows():
            if index % 2000 == 0:
                print(index)

            token = segment(row['text'])
            rough_res, rigour_res = clean_sen(token)

            sent_num1, max_sentlen1, doc_len1 = com_len(rough_res)
            sent_num2, max_sentlen2, doc_len2 = com_len(rigour_res)

            buffer.append({
                'id': row['id'],
                'text': row['text'],
                'token': token,
                'rough_cleaned' : rough_res,
                'rigour_cleaned' : rigour_res,
                'is_fact': row['is_fact'],
                'statute_code': str(int(row['statute_code'])),
                'cls': row['cls'],
                'sent_num_rough': sent_num1,
                'max_sentlen_rough': max_sentlen1,
                'doc_len_rough': doc_len1,
                'sent_num_rigour': sent_num2,
                'max_sentlen_rigour': max_sentlen2,
                'doc_len_rigour': doc_len2,
            })
            if len(buffer) >= 20000:
                col.insert_many(buffer)
                buffer.clear()
        if len(buffer) > 0:
            col.insert_many(buffer)
            buffer.clear()

        print('save db finished !!!')


    def gen_all_info_csv(self, all_info_path = '../data/all_info.csv'):
        if os.path.exists(all_info_path):
            all_info = pd.read_csv(all_info_path)
        else:
            col = self.conn[self.to_db][self.all_info_col]
            all_info = []
            i = 0
            for item in col.find():

                i += 1
                if i % 50000 == 0:
                    print('case_code: ' + str(i))

                all_info.append([item['id'], item['rough_cleaned'], item['rigour_cleaned'],
                                 item['is_fact'], item['statute_code'], item['cls'],
                                 item['sent_num_rough'], item['max_sentlen_rough'], item['doc_len_rough'],
                                 item['sent_num_rigour'], item['max_sentlen_rigour'], item['doc_len_rigour'],])

            all_info = pd.DataFrame(all_info, columns=['id', 'rough_token', 'rigour_token',
                                                       'is_fact', 'statute_code', 'cls',
                                                       'sent_num_rough', 'max_sentlen_rough', 'doc_len_rough',
                                                       'sent_num_rigour', 'max_sentlen_rigour', 'doc_len_rigour',])

            all_info.dropna(how='any', inplace=True)
            print('all info length : {}'.format(str(len(all_info))))
            print('all_info to csv...')
            all_info.to_csv(all_info_path, index=0)

        return all_info



if __name__ == '__main__':
    all_info = AllInfo()

    # all_info.gen_all_data_to_db()

    all_info_df = all_info.gen_all_info_csv()

    print('finished !!!')