#coding:utf-8

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import pandas as pd
import numpy as np
import os


class Doc2vec:
    def __init__(self, vec_num, min_count = 2):
        self.min_count = min_count
        self.vec_num = vec_num
        self.model = None

    def train(self, doc, batch_size=10000, epochs=20):
        corpus = [TaggedDocument(d, [i]) for i, d in enumerate(doc)]
        model = Doc2Vec(size=self.vec_num, min_count=self.min_count)
        model.build_vocab(corpus)
        n = len(corpus) // batch_size + 1
        for i in range(n):
            print('batch-{}'.format(i))
            start_index = batch_size * i
            if start_index > len(corpus) - 1:
                break
            docs = corpus[start_index: start_index + batch_size]
            model.train(docs, total_examples=model.corpus_count, epochs=epochs)
        self.model = model

    def load_model(self, path):
        if os.path.exists(path):
            self.model = Doc2Vec.load(path)
            print('model load success!')
        else:
            raise FileNotFoundError('"{}" file not found!'.format(path))

    def save(self, path):
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.model.save(path)
        print('model save success!')

    def get_topic_vec(self):
        matrix = self.model.docvecs.vectors_docs
        df = pd.DataFrame(matrix)
        return df

    def infer_vec(self, doc):
        if isinstance(doc, list):
            if isinstance(doc[0], list):
                pass
            else:
                doc = [doc]
        else:
            raise TypeError('input must be list')

        res = []
        for d in doc:
            res.append(self.model.infer_vector(d))
        return np.array(res)
