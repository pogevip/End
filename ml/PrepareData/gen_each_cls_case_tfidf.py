from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle, os


def load_stop_words(path = 'data/stopWords.txt'):
    stw = []
    with open(path, 'r') as fp:
        for line in fp:
            stw.append(line.strip())
    return stw

stop_words = load_stop_words()
stop_flags = ['b', 'c', 'e', 'h', 'k', 'l', 'o', 's', 'u', 'w', 'x', 'y', 'z', 'un']


def gen_tfidf_and_dic_data(df):

    sentences_train = df['token']

    count_vec = TfidfVectorizer()
    model = count_vec.fit(sentences_train)

    all_tfidf = model.transform(sentences_train).tocoo()
    dictionary = model.get_feature_names()

    tfidf_df = pd.DataFrame({'doc_index': all_tfidf.row,
                             'word_index': all_tfidf.col,
                             'tfidf': all_tfidf.data})

    dic = {}
    for index, word in enumerate(dictionary):
        dic[word] = index

    return tfidf_df, dic


def main(path):
    all_info = pd.read_csv('train_set/all_info.csv')

    cls_set = set(all_info['cls'].tolist())

    def clean_sen(sen):
        res = []
        for item in sen.split('。/x'):
            item = item.strip()
            r = []
            for x in item.split(' '):
                try:
                    word, flag = x.split('/')
                    if word not in stop_words and flag not in stop_flags:
                        r.append(word)
                except:
                    continue
            res.append(' '.join(r))
        return '。'.join(res)

    for cls in cls_set:
        tmp = all_info[all_info['cls'] == cls]
        print(cls, len(tmp))

        dir = os.path.join('../train_set/each_cls', str(cls))

        if not os.path.exists(dir):
            os.makedirs(dir)

        print('     sentence cleanning...')
        info_path = os.path.join(dir, 'info.csv')
        tmp['token'] = tmp['token'].apply(clean_sen)
        print('     writing info...')
        tmp.to_csv(info_path, index=0)

        tfidf_path = os.path.join(dir, 'tfidf.csv')
        dic_path = os.path.join(dir, 'dic.pkl')

        print('     generating tfidf and dic...')
        tfidf_df, dic = gen_tfidf_and_dic_data(tmp)

        print('     writing dic')
        with open(dic_path, 'wb') as fp:
            pickle.dump(dic, fp)

        print('     writing tfidf')
        tfidf_df.to_csv(tfidf_path, index=0)


if __name__ == '__main__':
    all_info_path = '../data/all_info.csv'
    main(all_info_path)