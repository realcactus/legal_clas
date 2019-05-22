#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'zhouxiaosong'

from collections import Counter
# from data.word2vec_helper import embedding_sentences, output_vocab
# 试试fasttext

import numpy as np
import tensorflow.contrib.keras as kr
import re
import io
import collections
from operator import itemgetter




def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')

# 按频率排序输出每个类别的keycharacters
def build_keywords(filename):
    """读取文件数据"""

    categories = ['机械制造行业', '五金建材行业', '农林牧渔行业',
                  '化工行业', '电子通讯行业', '文体生活用品行业',
                  '农副食品行业', '纺织服饰行业', '家电行业', '其他行业',
                  '食品药品行业', '交通运输行业', '酒水饮料茶奶行业']
    with open_file(filename) as f:
        # 机械制造行业
        counter_0 = collections.Counter()
        # 五金建材行业
        counter_1 = collections.Counter()
        # 农林牧渔行业
        counter_2 = collections.Counter()
        # 化工行业
        counter_3 = collections.Counter()
        # 电子通讯行业
        counter_4 = collections.Counter()
        # 文体生活用品行业
        counter_5 = collections.Counter()
        # 农副食品行业
        counter_6 = collections.Counter()
        # 纺织服饰行业
        counter_7 = collections.Counter()
        # 家电行业
        counter_8 = collections.Counter()
        # 酒水饮料茶奶行业
        counter_9 = collections.Counter()
        # 食品药品行业
        counter_10 = collections.Counter()
        # 交通运输行业
        counter_11 = collections.Counter()
        # 其他行业
        counter_12 = collections.Counter()

        for line in f:
            try:
                label, content = line.strip().split('\t')
                if label == categories[0]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_0[word] += 1
                elif label == categories[1]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_1[word] += 1
                elif label == categories[2]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_2[word] += 1
                elif label == categories[3]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_3[word] += 1
                elif label == categories[4]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_4[word] += 1
                elif label == categories[5]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_5[word] += 1
                elif label == categories[6]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_6[word] += 1
                elif label == categories[7]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_7[word] += 1
                elif label == categories[8]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_8[word] += 1
                elif label == categories[9]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_9[word] += 1
                elif label == categories[10]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_10[word] += 1
                elif label == categories[11]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_11[word] += 1
                elif label == categories[12]:
                    if content:
                        content = str(content).strip()
                        temp_list = list(content)
                        while ' ' in temp_list:
                            temp_list.remove(' ')
                        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．',
                                             ')', '(', '／', '％', '…', '［', '］']
                        for c in remove_characters:
                            while c in temp_list:
                                temp_list.remove(c)
                        for word in temp_list:
                            counter_12[word] += 1

            except:
                pass
    sorted_wotrd_to_cnt_class_0 = sorted(counter_0.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_0 = [x[0] for x in sorted_wotrd_to_cnt_class_0]

    sorted_wotrd_to_cnt_class_1 = sorted(counter_1.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_1 = [x[0] for x in sorted_wotrd_to_cnt_class_1]

    sorted_wotrd_to_cnt_class_2 = sorted(counter_2.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_2 = [x[0] for x in sorted_wotrd_to_cnt_class_2]

    sorted_wotrd_to_cnt_class_3 = sorted(counter_3.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_3 = [x[0] for x in sorted_wotrd_to_cnt_class_3]

    sorted_wotrd_to_cnt_class_4 = sorted(counter_4.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_4 = [x[0] for x in sorted_wotrd_to_cnt_class_4]

    sorted_wotrd_to_cnt_class_5 = sorted(counter_5.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_5 = [x[0] for x in sorted_wotrd_to_cnt_class_5]

    sorted_wotrd_to_cnt_class_6 = sorted(counter_6.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_6 = [x[0] for x in sorted_wotrd_to_cnt_class_6]

    sorted_wotrd_to_cnt_class_7 = sorted(counter_7.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_7 = [x[0] for x in sorted_wotrd_to_cnt_class_7]

    sorted_wotrd_to_cnt_class_8 = sorted(counter_8.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_8 = [x[0] for x in sorted_wotrd_to_cnt_class_8]

    sorted_wotrd_to_cnt_class_9 = sorted(counter_9.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_9 = [x[0] for x in sorted_wotrd_to_cnt_class_9]

    sorted_wotrd_to_cnt_class_10 = sorted(counter_10.items(),
                                         key=itemgetter(1),
                                         reverse=True)
    sorted_words_class_10 = [x[0] for x in sorted_wotrd_to_cnt_class_10]

    sorted_wotrd_to_cnt_class_11 = sorted(counter_11.items(),
                                          key=itemgetter(1),
                                          reverse=True)
    sorted_words_class_11 = [x[0] for x in sorted_wotrd_to_cnt_class_11]

    sorted_wotrd_to_cnt_class_12 = sorted(counter_12.items(),
                                          key=itemgetter(1),
                                          reverse=True)
    sorted_words_class_12 = [x[0] for x in sorted_wotrd_to_cnt_class_12]

    # 统计每一个类别出现次数最多的词
    # 还有一个操作，对比13个类别中的所有的词，将每一个类别都会出现的词删去，只保留该类别特定的词

    # sorted_words_class_0_bak = sorted_words_class_0[:]
    # sorted_words_class_1_bak = sorted_words_class_1[:]
    # sorted_words_class_2_bak = sorted_words_class_2[:]
    # sorted_words_class_3_bak = sorted_words_class_3[:]
    # sorted_words_class_4_bak = sorted_words_class_4[:]
    # sorted_words_class_5_bak = sorted_words_class_5[:]
    # sorted_words_class_6_bak = sorted_words_class_6[:]
    # sorted_words_class_7_bak = sorted_words_class_7[:]
    # sorted_words_class_8_bak = sorted_words_class_8[:]
    # sorted_words_class_9_bak = sorted_words_class_9[:]
    # sorted_words_class_10_bak = sorted_words_class_10[:]
    # sorted_words_class_11_bak = sorted_words_class_11[:]
    # sorted_words_class_12_bak = sorted_words_class_12[:]
    # for word in sorted_words_class_0_bak:
    #     if word in sorted_words_class_1_bak or word in sorted_words_class_2_bak\
    #             or word in sorted_words_class_3_bak or word in sorted_words_class_4_bak \
    #             or word in sorted_words_class_5_bak or word in sorted_words_class_6_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_0.remove(word)
    #
    # for word in sorted_words_class_1_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_2_bak\
    #             or word in sorted_words_class_3_bak or word in sorted_words_class_4_bak \
    #             or word in sorted_words_class_5_bak or word in sorted_words_class_6_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_1.remove(word)
    #
    # for word in sorted_words_class_2_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_3_bak or word in sorted_words_class_4_bak \
    #             or word in sorted_words_class_5_bak or word in sorted_words_class_6_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_2.remove(word)
    #
    # for word in sorted_words_class_3_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_4_bak \
    #             or word in sorted_words_class_5_bak or word in sorted_words_class_6_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_3.remove(word)
    #
    # for word in sorted_words_class_4_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_5_bak or word in sorted_words_class_6_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_4.remove(word)
    #
    # for word in sorted_words_class_5_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_6_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_5.remove(word)
    #
    # for word in sorted_words_class_6_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_7_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_6.remove(word)
    #
    # for word in sorted_words_class_7_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_6_bak or word in sorted_words_class_8_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_7.remove(word)
    #
    # for word in sorted_words_class_8_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_6_bak or word in sorted_words_class_7_bak \
    #             or word in sorted_words_class_9_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_8.remove(word)
    #
    # for word in sorted_words_class_9_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_6_bak or word in sorted_words_class_7_bak \
    #             or word in sorted_words_class_8_bak or word in sorted_words_class_10_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_9.remove(word)
    #
    # for word in sorted_words_class_10_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_6_bak or word in sorted_words_class_7_bak \
    #             or word in sorted_words_class_8_bak or word in sorted_words_class_9_bak \
    #             or word in sorted_words_class_11_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_10.remove(word)
    #
    # for word in sorted_words_class_11_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_6_bak or word in sorted_words_class_7_bak \
    #             or word in sorted_words_class_8_bak or word in sorted_words_class_9_bak \
    #             or word in sorted_words_class_10_bak or word in sorted_words_class_12_bak:
    #         sorted_words_class_11.remove(word)
    #
    # for word in sorted_words_class_12_bak:
    #     if word in sorted_words_class_0_bak or word in sorted_words_class_1_bak\
    #             or word in sorted_words_class_2_bak or word in sorted_words_class_3_bak \
    #             or word in sorted_words_class_4_bak or word in sorted_words_class_5_bak \
    #             or word in sorted_words_class_6_bak or word in sorted_words_class_7_bak \
    #             or word in sorted_words_class_8_bak or word in sorted_words_class_9_bak \
    #             or word in sorted_words_class_10_bak or word in sorted_words_class_11_bak:
    #         sorted_words_class_12.remove(word)

    with open('usefuldata-711depart/keycharacters/机械制造行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_0:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/五金建材行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_1:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/农林牧渔行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_2:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/化工行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_3:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/电子通讯行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_4:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/文体生活用品行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_5:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/农副食品行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_6:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/纺织服饰行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_7:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/家电行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_8:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/其他行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_9:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/食品药品行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_10:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/交通运输行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_11:
            f.write(word)
            f.write('\n')
    with open('usefuldata-711depart/keycharacters/酒水饮料茶奶行业.txt', 'w', encoding='utf-8') as f:
        for word in sorted_words_class_12:
            f.write(word)
            f.write('\n')

def read_category():
    """读取分类目录"""
    categories = ['机械制造行业', '五金建材行业', '农林牧渔行业',
                  '化工行业', '电子通讯行业', '文体生活用品行业',
                  '农副食品行业', '纺织服饰行业', '家电行业', '其他行业',
                  '食品药品行业', '交通运输行业', '酒水饮料茶奶行业']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

if __name__ == '__main__':
    build_keywords('usefuldata-711depart/train.txt')