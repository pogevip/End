#coding:utf-8

from gensim import corpora, matutils
from gensim.models import LsiModel, LdaModel, TfidfModel
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import pandas as pd
import os
from abc import ABCMeta, abstractmethod
import numpy as np


class DocVec(metaclass=ABCMeta):
    def __init__(self, vec_num):
        self.vec_num = vec_num

    def fit(self, doc, out_dir):
        self.doc = doc
        self.out_dir = out_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    @abstractmethod
    def get_topic_vec(self):
        '''
        :return:
        '''

    @abstractmethod
    def infer_vec(self, doc):
        '''
        :return:
        '''



class TopicVec(DocVec):
    def __init__(self, vec_num):
        DocVec.__init__(self, vec_num)
        self.dictionary = None
        self.tfidf_model = None
        self.model = None

    def __gen_dictionary(self, doc):
        dictionary = corpora.Dictionary(doc)
        dictionary.save(os.path.join(self.out_dir, 'dict.dict'))
        return dictionary

    def __get_dictionary(self):
        if os.path.exists(os.path.join(self.out_dir, 'dict.dict')):
            dictionary = corpora.Dictionary.load(os.path.join(self.out_dir, 'dict.dict'))
        else:
            raise FileNotFoundError('"dict.dict" file not found!')
        return dictionary

    def __gen_tfidf_model(self, corpus):
        tfidf_model = TfidfModel(corpus)
        tfidf_model.save(os.path.join(self.out_dir, 'tfidf.model'))
        return tfidf_model

    def __get_tfidf_model(self):
        if os.path.exists(os.path.join(self.out_dir, 'tfidf.model')):
            tfidf_model = TfidfModel.load(os.path.join(self.out_dir, 'tfidf.model'))
        else:
            raise FileNotFoundError('"tfidf.model" file not found!')
        return tfidf_model

    def __gen_model(self, corpus):
        pass

    def __get_model(self):
        pass

    def fit(self, doc, out_dir, use_exist_dictionary):
        DocVec.fit(self, doc, out_dir)

        if use_exist_dictionary:
            if not self.dictionary:
                self.dictionary = self.__get_dictionary()
            if not self.tfidf_model:
                self.tfidf_model = self.__get_tfidf_model()
            corpus = [self.dictionary.doc2bow(d) for d in doc]
            self.corpus = self.tfidf_model[corpus]
        else:
            self.dictionary = self.__gen_dictionary(doc)
            corpus = [self.dictionary.doc2bow(d) for d in doc]
            self.tfidf_model = self.__gen_tfidf_model(corpus)
            self.corpus = self.tfidf_model[corpus]

    def get_topic_vec(self):
        matrix = matutils.corpus2dense(self.model[self.corpus], num_terms=self.vec_num).T
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

        if not self.dictionary:
            self.dictionary = self.__get_dictionary()
        if not self.tfidf_model:
            self.tfidf_model = self.__get_tfidf_model()
        corpus = [self.dictionary.doc2bow(d) for d in doc]
        corpus = self.tfidf_model[corpus]
        if not self.model:
            self.model = self.__get_model()
        return matutils.corpus2dense(self.model[corpus], num_terms=len(self.model.get_topics())).T


class LsiVec(TopicVec):
    def __init__(self, vec_num):
        TopicVec.__init__(self, vec_num)

    def __gen_model(self, corpus):
        # if self.p_corpus == 'onehot':
        #     model_name = 'lsi_one_hot.model'
        # else:
        #     model_name = 'lsi_tfidf.model'
        model_name = 'lsi.model'
        self.model = LsiModel(corpus, id2word=self.dictionary, num_topics=self.vec_num)
        self.model.save(os.path.join(self.out_dir, model_name))

    def __get_model(self):
        model_name = 'lsi.model'
        if os.path.exists(os.path.join(self.out_dir, model_name)):
            self.model = LsiModel.load(os.path.join(self.out_dir, model_name))
        else:
            raise FileNotFoundError('"{}" file not found!'.format(model_name))

    def fit(self, doc, out_dir, use_exist_dictionary = False):
        TopicVec.fit(self, doc, out_dir, use_exist_dictionary)
        self.__gen_model(self.corpus)



class LdaVec(TopicVec):
    def __init__(self, vec_num):
        TopicVec.__init__(self, vec_num)

    def __gen_model(self, corpus):
        # if self.p_corpus == 'onehot':
        #     model_name = 'lda_one_hot.model'
        # else:
        #     model_name = 'lda_tfidf.model'
        model_name = 'lda.model'
        self.model = LdaModel(corpus, id2word=self.dictionary, num_topics=self.vec_num)
        self.model.save(os.path.join(self.out_dir, model_name))

    def __get_model(self):
        model_name = 'lda.model'
        if os.path.exists(os.path.join(self.out_dir, model_name)):
            self.model = LdaModel.load(os.path.join(self.out_dir, model_name))
        else:
            raise FileNotFoundError('"{}" file not found!'.format(model_name))

    def fit(self, doc, out_dir, use_exist_dictionary = False):
        TopicVec.fit(self, doc, out_dir, use_exist_dictionary)
        self.__gen_model(self.corpus)



class Doc2vecVec(DocVec):
    def __init__(self, vec_num, min_count = 2):
        DocVec.__init__(self, vec_num)
        self.min_count = min_count
        self.model = None

    def __gen_model(self, corpus, batch_size, epochs):
        model_name = 'doc2vec.model'
        model = Doc2Vec(size=self.vec_num, min_count=self.min_count)
        model.build_vocab(corpus)
        n = len(corpus)//batch_size+1
        for i in range(n):
            print('batch-{}'.format(i))
            start_index = batch_size*i
            if start_index > len(corpus)-1:
                break
            docs = corpus[start_index: start_index+batch_size]
            model.train(docs, total_examples=model.corpus_count, epochs=epochs)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model.save(os.path.join(self.out_dir, model_name))
        return model

    def __get_model(self):
        model_name = 'doc2vec.model'
        if os.path.exists(os.path.join(self.out_dir, model_name)):
            return Doc2Vec.load(os.path.join(self.out_dir, model_name))
        else:
            raise FileNotFoundError('"{}" file not found!'.format(model_name))

    def fit(self, doc, out_dir, batch_size=10000, epochs=20):
        DocVec.fit(self, doc, out_dir)
        docs = [TaggedDocument(d, [i]) for i, d in enumerate(doc)]
        self.model = self.__gen_model(docs, batch_size, epochs)

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
        if not self.model:
            self.model = self.__get_model()
        res = []
        for d in doc:
            res.append(self.model.infer_vector(d))
        return np.array(res)


if __name__ == '__main__':
    pass
