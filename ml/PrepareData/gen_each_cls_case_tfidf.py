from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle, os


def gen_tfidf_abd_dic_data(data_path):
    tfidf_path = os.path.join(data_path.split('.')[0] + '_tfidf', 'csv')
    dic_path = os.path.join(data_path.split('.')[0] + '_dic', 'pkl')

    df = pd.read_csv(data_path)
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
    with open(dic_path, 'wb') as fp:
        pickle.dump(dic, fp)

    tfidf_df.to_csv(tfidf_path, index=0)