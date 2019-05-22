#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'zhouxiaosong'
import os
import sys
import logging
import multiprocessing
import time
import json
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

def output_vocab(vocab):
    for k, v in vocab.items():
        print(k)

def embedding_sentences(sentences, embedding_size = 100, window = 5, min_count = 5, file_to_load = None, file_to_save = None):
    '''
    embeding_size 词嵌入维数
    window : 上下文窗口
    min_count : 词频少于min_count会被删除
    '''
    if file_to_load is not None:
        # w2vModel = Word2Vec.load(file_to_load)
        # w2vModel = Word2Vec.wv.load_word2vec_format(file_to_load)
        w2vModel = KeyedVectors.load_word2vec_format(file_to_load)
        # print(w2vModel.most_similar("死亡"))
        print(w2vModel.most_similar("love"))
        print(w2vModel.most_similar("hate"))
        # print(w2vModel.most_similar("上市"))
        # print(w2vModel.most_similar("开始"))
        # print(w2vModel.most_similar("消费"))
        # print(w2vModel.most_similar("运动鞋"))
        # print(w2vModel.most_similar("皮鞋"))
    else:
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
        if file_to_save is not None:
            # w2vModel.save(file_to_save)
            w2vModel.wv.save_word2vec_format(file_to_save, binary=False)

    all_vectors = []
    embeddingDim = w2vModel.vector_size
    # 嵌入维数
    embeddingUnknown = [np.random.random() for i in range(embeddingDim)]
    print(embeddingUnknown)
    # 用字典，方便输出词汇
    # outtext = open('word2vec/character/character2vec_out_vocab.txt','w',encoding='utf-8')
    outtext = open('word2vec/word2vec-cbow-imdb/word2vec_out_vocab.txt', 'w', encoding='utf-8')
    for k in w2vModel.wv.vocab:
        outtext.write(k + '\n')
    outtext.close()
    word2vec_dict = {}
    for sentence in sentences:
        # this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                # this_vector.append(w2vModel[word])
                word2vec_dict[word] = w2vModel[word]
            else:
                # this_vector.append(embeddingUnknown)
                word2vec_dict[word] = embeddingUnknown
        # all_vectors.append(this_vector)
    word2vec_dict['<unk>'] = w2vModel['<unk>']
    return word2vec_dict


def get_word2vec_dict(sentences, file_to_load = None):
    w2vModel = KeyedVectors.load_word2vec_format(file_to_load)
    # print(w2vModel.most_similar("死亡"))
    print(w2vModel.most_similar("肇事"))
    word2vec_dict = {}
    for word in w2vModel.wv.vocab:
        word2vec_dict[word] = w2vModel[word]

    embeddingDim = w2vModel.vector_size
    # 嵌入维数
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        for word in sentence:
            if word not in w2vModel.wv.vocab:
                word2vec_dict[word] = embeddingUnknown
    return word2vec_dict

# def generate_word2vec_files(input_file, output_model_file, output_vector_file, size = 100, window = 5, min_count = 5):
#     start_time = time.time()
#
#     # trim unneeded model memory = use(much) less RAM
#     # model.init_sims(replace=True)
#     model = Word2Vec(LineSentence(input_file), size = size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
#     model.save(output_model_file)
#     model.wv.save_word2vec_format(output_vector_file, binary=False)
#
#     end_time = time.time()
#     print("used time : %d s" % (end_time - start_time))


# if __name__ == '__main__':
#
#
#
