#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'zhouxiaosong'

from collections import Counter
from data.word2vec_helper import embedding_sentences, output_vocab
# 试试fasttext
# from data.fasttextvec_helper import embedding_sentences

import numpy as np
import tensorflow.contrib.keras as kr
import jieba
import re
import io




def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8')


def read_file(filename):
    """读取文件数据"""

    # 用类别独特关键词过滤一下试试看？
    # ********************************************************************
    keyword_filter_file = open_file('data/keyword_filter.txt')
    keyword_filter = []
    index_tmp = 0
    for line in keyword_filter_file:
        index_tmp += 1
        keyword_filter.append(line.strip())
    print(index_tmp)
    keyword_filter_file.close()
    # *********************************************************************
    # 停用辞典

    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    # content是一段文本的内容，原本基于字符级的应该是直接list(content)就行了
                    # 现在打算尝试基于词语的
                    # 去除前后空格
                    content = str(content).strip()
                    content = str(content).lower()
                    # content = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;：． :／\-［］％。？、~@#￥%……&*（）]+", "", content)
                    # *************************************************************
                    # 中文
                    result = jieba.cut(content)
                    # *************************************************************
                    # *************************************************************
                    # 英文
                    # result = content.split()
                    # *************************************************************
                    # contents.append(list(str(content).lower()))
                    temp_list = list(result)
                    while ' ' in temp_list:
                        temp_list.remove(' ')
                    while '' in temp_list:
                        temp_list.remove('')

                    # 版本更新，这个过滤应该独立到预处理部分去做
                    # remove_characters = ['0','1','2','3','4','5','6','7','8','9','，','。','、','：','．',')','(','／','％','…','［', '］',
                    #                      '$','*','【','】','？','”','，','。','”','）','（','；','《','》','-']
                    remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                                         '(', '／', '％', '…', '［', '］']
                    for c in remove_characters:
                        while c in temp_list:
                            temp_list.remove(c)
                    # 用keyword过滤
                    # *************************************************
                    # content_list = []
                    # for word in temp_list:
                    #     if word in keyword_filter:
                    #         content_list.append(word)
                    # *************************************************

                    contents.append(temp_list)
                    # contents.append(content_list)

                    labels.append(label)
            except:
                pass
    return contents, labels


def get_maxlength(filename):
    max_value = 0
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:

                    content = str(content).strip()
                    content = str(content).lower()
                    # content = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;：． :／\-［］％。？、~@#￥%……&*（）]+", "",
                    #                  content)
                    result = jieba.cut(content)
                    temp_list = list(result)
                    while ' ' in temp_list:
                        temp_list.remove(' ')
                    # remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                    #                      '(', '／', '％', '…', '［', '］',
                    #                      '$', '*', '【', '】', '？', '”', '，', '。', '“', '）', '（', '；', '《', '》', '-']
                    remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                                         '(', '／', '％', '…', '［', '］']
                    for c in remove_characters:
                        while c in temp_list:
                            temp_list.remove(c)
                    max_value = max(max_value, len(temp_list))
            except:
                pass
    return max_value


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    # 这个就是list形式的sentence
    # word2vec_dict = embedding_sentences(data_train, file_to_save='word2vec/word2vec.model')
    # 试试fasttext_vec
    # word2vec_dict = embedding_sentences(data_train, file_to_save='word2vec/word2vec.txt')

    embedding_sentences(data_train, file_to_save='word2vec/word2vec-cbow-imdb/word2vec.txt')

    # out_file = open('word2vec/out_key_value.txt','w',encoding='utf-8')
    # out_file = open('word2vec/fasttext_vec/out_key_value.txt', 'w', encoding='utf-8')
    # for k, v in word2vec_dict.items():
    #     out_file.write(k)
    #     out_file.write(str(v) + '\n')
    # out_file.close()

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    print(counter)
    count_pairs = counter.most_common(46096)
    # count_pairs = counter.most_common(25019)
    print(count_pairs)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<UNK>'] + list(words)
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')



def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        words = [(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录"""
    categories = ['机械制造行业', '五金建材行业', '农林牧渔行业',
                  '化工行业', '电子通讯行业', '文体生活用品行业',
                  '农副食品行业', '纺织服饰行业', '家电行业', '其他行业',
                  '食品药品行业', '交通运输行业', '酒水饮料茶奶行业']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        # 这里我认为空格对于英文句子语义来说同样是很重要的
        temp = []
        for x in contents[i]:
            if x in word_to_id:
                temp.append(word_to_id[x])
            else:
                temp.append(word_to_id['<UNK>'])
        data_id.append(temp)
        # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # print(data_id[0])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad
    # return data_id, label_id

def create_corpus(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(filename)
    file_out_corpus = open('corpus.txt', 'w', encoding='utf-8')
    for i in range(len(contents)):
        for x in contents[i]:
            if x in word_to_id:
                file_out_corpus.write(x)
                file_out_corpus.write(' ')
        file_out_corpus.write('\n')


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = []
    # for index in indices:
    #     x_shuffle.append(x[index])
    #     # try:
    #     #     x_shuffle[i] = x[index]
    #     # except:
    #     #     print(i)
    # y_shuffle = []
    # for index_j in indices:
    #     # y_shuffle[j] = y[index_j]
    #     y_shuffle.append(y[index_j])

    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
