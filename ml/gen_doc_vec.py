import pandas as pd
from docvec.doc2vec import Doc2vec
import os

print('reading...')
data_path = 'data/trainSet/train_info.csv'
df = pd.read_csv(data_path)

print('preprocess..')
df = df.loc[:,['rigour_token', 'cls', 'is_gen', 'train_val_test']]
df = df[(df['train_val_test']==1) | (df['train_val_test']==2)]
df.dropna(inplace=True, how='any')
df.reset_index()
df['token'] = df['rigour_token'].apply(lambda x:x.replace('。', ' ').split(' '))

text = df['token'].tolist()
label = df.loc[:,['cls', 'is_gen', 'train_val_test']]

# lda = LdaVec(100)
# lda.fit(text, 'data/trainSet/docvec1', use_exist_dictionary=False)
# vec = lda.get_topic_vec()
# res = pd.concat([vec, label], axis=1)
# res.to_csv('data/trainSet/docvec1/lda.csv', index=0)

print('model...')
d2v = Doc2vec(300)
d2v.train(text)
dir = 'data/trainSet/docvec'
if not os.path.exists(dir):
    os.makedirs(dir)
d2v.save(os.path.join(dir, 'd2v.model'))

# vec = d2v.get_topic_vec()
# res = pd.concat([vec, label], axis=1)
# print('saving...')
# res.to_csv('data/trainSet/docvec/d2v.csv', index=0)
# print('finished!')

# lsi = LsiVec(256)
# lsi.fit(text, 'data/trainSet/docvec1', use_exist_dictionary=False)
# vec = lsi.get_topic_vec()
# res = pd.concat([vec, label], axis=1)
# res.to_csv('data/trainSet/docvec1/lsi.csv', index=0)
