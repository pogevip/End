import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

import pickle, os, random, math

CLS = ['9001', '9012', '9047', '9130', '9299',
       '9461', '9483', '9542', '9705', '9771']


def random_delete_word(word, threshold=0.15):
    if random.random() >= threshold:
        return word


def process_token(doc):
    doc = doc.split(' ')
    doc = doc if len(doc) <= 200 else doc[:200]
    doc = list(filter(lambda x: x, map(random_delete_word, doc)))
    return ' '.join(doc)


def gen_lda_data(src_path, out_path, min_num=20000):
    res = pd.DataFrame([], columns=['token'])
    for c in CLS:
        print(c)
        data = pd.read_csv(os.path.join(src_path, c + '.csv'))
        print(len(data))
        if len(data) < min_num:
            tmp = data[data['lenth'] >= 20]
            buffer = []
            multiple = int(math.ceil(min_num / len(tmp)))
            for _, row in data.iterrows():
                buffer.append([row['token']])
                for i in range(multiple):
                    token = process_token(row['token'])
                    buffer.append([token])
            df = pd.DataFrame(buffer, columns=['token'])
            df = df.sample(min_num)
            print(len(df))
            res = pd.concat([res, df], ignore_index=True)
        else:
            df = data.loc[:, ['token']]
            print(len(df))
            res = pd.concat([res, df], ignore_index=True)
    res['token'] = res['token'].apply(lambda x: x.replace('。', ' ').split(' '))
    res = res['token'].tolist()
    print(len(res))
    with open(os.path.join(out_path, 'train_lda_model_data.pkl'), 'wb') as fp:
        pickle.dump(res, fp)

    return res


def train_lda_model(src, num_topic, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print('load data...')
    if isinstance(src, str):
        with open(os.path.join(src, 'train_lda_model_data.pkl'), 'rb') as fp:
            src_data = pickle.load(fp)
    else:
        src_data = src

    print('dic...')
    dictionary = Dictionary(src_data)
    print('corpus...')
    corpus = [dictionary.doc2bow(text) for text in src_data]
    print('training...')
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic)

    print(lda.get_document_topics(corpus[0]))
    print('save...')
    dictionary.save(os.path.join(model_path, 'lda.dic'))
    lda.save(os.path.join(model_path, 'lda.model'))

    print('finished!')


def load_cls_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    dic = {cls: item[0] for cls, item in dic.items()}
    return dic


def gen_web_train_data(src_path, out_path, min_num=20000):
    res = pd.DataFrame([], columns=['token', 'cls', 'is_gen'])
    for c in CLS:
        print(c)
        data = pd.read_csv(os.path.join(src_path, c + '.csv'))
        print(len(data))
        if len(data) < min_num:
            tmp = data[data['lenth'] >= 20]
            buffer = []
            multiple = int(math.ceil(min_num / len(tmp)))
            for _, row in data.iterrows():
                buffer.append([row['token'], row['cls'], 0])
                for i in range(multiple):
                    token = process_token(row['token'])
                    buffer.append([token, row['cls'], 1])
            df = pd.DataFrame(buffer, columns=['token', 'cls', 'is_gen'])
            df = df.sample(min_num)
            print(len(df))
            res = pd.concat([res, df], ignore_index=True)
        else:
            df = data.loc[:, ['token', 'cls']]
            df['is_gen'] = 0
            print(len(df))
            res = pd.concat([res, df], ignore_index=True)
    
    res['train_val_test'] = 1
    # 划分测试集
    for _, group in res.groupby('cls'):
        #             num = (math.ceil((len(group)//10000+0.1)*0.2)-1)*10000
        test_index = group.sample(frac=0.2).index
        res.loc[test_index, 'train_val_test'] = 3

    # 划分验证集
    test_df = res[res['train_val_test'] == 3]
    for _, group in test_df.groupby('cls'):
        val_index = group.sample(frac=0.6).index
        res.loc[val_index, 'train_val_test'] = 2

    res.to_csv(out_path, index=0)
    print('train data finished!')

def load_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    return dic

def gen_train_file(model_path, src_path, cls_dic_path, word_dict_path, out_path):
    dictionary = Dictionary.load(os.path.join(model_path, 'lda.dic'))
    model = LdaModel.load(os.path.join(model_path, 'lda.model'), mmap='r')
    
#     word_dict = load_dict(word_dict_path)
    
    cls_dic = load_cls_dict(cls_dic_path)
    data = pd.read_csv(src_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

#     data['X'] = data['token'].apply(lambda doc:
#                                             list(map(lambda w: word_dict[w] if w in word_dict else 0, doc.replace('。', ' ').split(' '))))
    data['token'] = data['token'].apply(lambda doc : doc.replace('。', ' ').split(' '))
    data['Y'] = data['cls'].apply(lambda cls: cls_dic[cls])

    for tag, group in data.groupby('train_val_test'):
        print(tag)
        if tag == 1:
            out = 'train.pkl'
        elif tag == 2:
            out = 'val.pkl'
        elif tag == 3:
            out = 'test.pkl'
        else:
            raise ValueError('tag error!')
        
        tmp = group.sample(frac=1).reset_index(drop=True)
        
#         X = tmp['X'].tolist()

        docs = tmp['token'].tolist()
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        X = list(model.get_document_topics(corpus))
        
        Y = tmp['Y'].tolist()

        res = {
            'X': X,
            'Y': Y
        }

        with open(os.path.join(out_path, out), 'wb') as fp:
            pickle.dump(res, fp)

        print('finish!')


if __name__ == '__main__':
    src_path = '../data/trainSet/rank/web_each_cls_data/'
    out_path = '../data/trainSet/classifier/'

    #     lda_data = gen_lda_data(src_path, out_path)

    lda_model_path = '../data/trainSet/classifier/lda/'
    train_lda_model(out_path, 40, lda_model_path)
    
#     train_data_path = '../data/trainSet/classifier/web_train_info.csv'
#     gen_web_train_data(src_path, out_path=train_data_path)

#     cls_dict_path = '../data/trainSet/classifier/cls_5w.dic'
#     word_dict_path = '../data/trainSet/classifier/dictionary.dic'
#     train_file_lda_path = '../data/trainSet/classifier/train_file_web_lda/'
#     gen_train_file(model_path=lda_model_path, 
#                    src_path=train_data_path, 
#                    cls_dic_path=cls_dict_path, 
#                    word_dict_path=word_dict_path,
#                    out_path=train_file_lda_path)
