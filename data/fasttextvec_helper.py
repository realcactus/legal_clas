#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing

from gensim.models import FastText
from gensim.models import KeyedVectors


# 目前这个是无法在线更新训练的，貌似
def embedding_sentences(sentences, embedding_size=100, window=5, min_count=5, file_to_load=None, file_to_save=None):
    '''
    embeding_size 词嵌入维数
    window : 上下文窗口
    min_count : 词频少于min_count会被删除
    '''
    if file_to_load is not None:
        # w2vModel = Word2Vec.load(file_to_load)
        w2vModel = KeyedVectors.load_word2vec_format(file_to_load)
        # print(w2vModel.most_similar("死亡"))
        print(w2vModel.most_similar("死亡"))
    else:
        w2vModel = FastText(sentences, size = embedding_size, window = window, min_count = min_count,
                            workers = multiprocessing.cpu_count(), iter=30, min_n=2, max_n=6, word_ngrams=1)
        if file_to_save is not None:
            # w2vModel.save(file_to_save)
            w2vModel.wv.save_word2vec_format(file_to_save, binary=False)

    all_vectors = []
    embeddingDim = w2vModel.vector_size
    # 嵌入维数
    embeddingUnknown = [0 for i in range(embeddingDim)]
    # 用字典，方便输出词汇
    # outtext = open('word2vec/character/character2vec_out_vocab.txt','w',encoding='utf-8')
    outtext = open('word2vec/fasttext_vec/word2vec_out_vocab.txt', 'w', encoding='utf-8')
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
    return word2vec_dict