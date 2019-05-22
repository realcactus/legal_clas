#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

import os
import csv
import pandas as pd
import numpy as np
from data import data_loader
from data import data_loader_wordlevel
from cnn_model_2input import TCNNConfig
from sklearn.model_selection import KFold,StratifiedKFold

base_dir = 'data/imdb'
# train_dir = os.path.join(base_dir, 'train.txt')
# val_dir = os.path.join(base_dir, 'val.txt')
# test_dir = os.path.join(base_dir, 'test.txt')
all_data_dir = os.path.join(base_dir, 'all.txt')
vocab_dir_c = os.path.join(base_dir, 'vocab-c.txt')
vocab_dir_w = os.path.join(base_dir, 'vocab-w.txt')


def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')


def save_file(filename, ssdfile_dir):
    """
    将文件分流到3个文件中
    filename: 原数据地址，一个csv文件
    文件内容格式:  类别\t内容
    """
    f_train = open('../data/usefuldata-37depart/train.txt', 'w', encoding='utf-8')
    # f_val = open('../data/usefuldata-37depart/val.txt', 'w', encoding='utf-8')
    f_test = open('../data/usefuldata-37depart/test.txt', 'w', encoding='utf-8')
    f_class = open('../data/usefuldata-37depart/class.txt', 'w', encoding='utf-8')

    # 事先将事实段全文和文件名之间建立一个字典
    dict_ssdqw = {}
    for ssdfile in os.listdir(ssdfile_dir):
        ssdfile_name = os.path.join(ssdfile_dir, ssdfile)
        f = open(ssdfile_name, 'r', encoding='utf-8')
        content_qw = ''
        content = f.readline()
        # 以下部分，因为统计整个案件基本情况他有换行，所以将多行处理在一行里面
        while content:
            content_qw += content
            content_qw = content_qw.replace('\n', '')
            content = f.readline()
        ssdfile_key = str(ssdfile).replace('.txt','')
        dict_ssdqw[ssdfile_key] = content_qw

    # doc_count代表数据总共有多少行
    doc_count = 0
    temp_file = open(filename, 'r', encoding='utf-8')
    line = temp_file.readline()
    while line:
        doc_count += 1
        line = temp_file.readline()
    temp_file.close()

    class_set = set()
    # tag_train = doc_count * 8 / 10
    # tag_val = doc_count * 9 / 10
    tag_test = doc_count * 70 / 100
    tag = 0
    # 有些文书行业标记是空！！我想看看有多少条？
    blank_tag = 0
    # 标记一下，每个类别有多少个训练集、验证集、测试集？
    train_class_tag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    val_class_tag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_class_tag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # csvfile = open(filename, 'r', encoding='utf-8')
    txtfile = open(filename, 'r', encoding='utf-8')
    process_line = txtfile.readline()
    while process_line:
        tag += 1
        line_content = process_line.split(',')
        name = line_content[0]
        if name in dict_ssdqw:
            content = dict_ssdqw[name]
            label = line_content[1]
            # if label != '' and label != '其他行业':
            if label != '':
                class_set.add(label)
                if tag < tag_test:
                    f_train.write(label + '\t' + content + '\n')
                    if label == '机械制造行业':
                        train_class_tag[0] += 1
                    elif label == '五金建材行业':
                        train_class_tag[1] += 1
                    elif label == '农林牧渔行业':
                        train_class_tag[2] += 1
                    elif label == '化工行业':
                        train_class_tag[3] += 1
                    elif label == '电子通讯行业':
                        train_class_tag[4] += 1
                    elif label == '文体生活用品行业':
                        train_class_tag[5] += 1
                    elif label == '农副食品行业':
                        train_class_tag[6] += 1
                    elif label == '纺织服饰行业':
                        train_class_tag[7] += 1
                    elif label == '家电行业':
                        train_class_tag[8] += 1
                    elif label == '其他行业':
                        train_class_tag[9] += 1
                    elif label == '食品药品行业':
                        train_class_tag[10] += 1
                    elif label == '交通运输行业':
                        train_class_tag[11] += 1
                    elif label == '酒水饮料茶奶行业':
                        train_class_tag[12] += 1
                # elif tag < tag_val:
                else:
                    # f_val.write(label + '\t' + content + '\n')
                    f_test.write(label + '\t' + content + '\n')
                    if label == '机械制造行业':
                        val_class_tag[0] += 1
                    elif label == '五金建材行业':
                        val_class_tag[1] += 1
                    elif label == '农林牧渔行业':
                        val_class_tag[2] += 1
                    elif label == '化工行业':
                        val_class_tag[3] += 1
                    elif label == '电子通讯行业':
                        val_class_tag[4] += 1
                    elif label == '文体生活用品行业':
                        val_class_tag[5] += 1
                    elif label == '农副食品行业':
                        val_class_tag[6] += 1
                    elif label == '纺织服饰行业':
                        val_class_tag[7] += 1
                    elif label == '家电行业':
                        val_class_tag[8] += 1
                    elif label == '其他行业':
                        val_class_tag[9] += 1
                    elif label == '食品药品行业':
                        val_class_tag[10] += 1
                    elif label == '交通运输行业':
                        val_class_tag[11] += 1
                    elif label == '酒水饮料茶奶行业':
                        val_class_tag[12] += 1
                # else:
                #     f_test.write(label + '\t' + content + '\n')
                #     if label == '机械制造行业':
                #         test_class_tag[0] += 1
                #     elif label == '五金建材行业':
                #         test_class_tag[1] += 1
                #     elif label == '农林牧渔行业':
                #         test_class_tag[2] += 1
                #     elif label == '化工行业':
                #         test_class_tag[3] += 1
                #     elif label == '电子通讯行业':
                #         test_class_tag[4] += 1
                #     elif label == '文体生活用品行业':
                #         test_class_tag[5] += 1
                #     elif label == '农副食品行业':
                #         test_class_tag[6] += 1
                #     elif label == '纺织服饰行业':
                #         test_class_tag[7] += 1
                #     elif label == '家电行业':
                #         test_class_tag[8] += 1
                #     elif label == '其他行业':
                #         test_class_tag[9] += 1
                #     elif label == '食品药品行业':
                #         test_class_tag[10] += 1
                #     elif label == '交通运输行业':
                #         test_class_tag[11] += 1
                #     elif label == '酒水饮料茶奶行业':
                #         test_class_tag[12] += 1
            else:
                blank_tag += 1
        process_line = txtfile.readline()

    for i in class_set:
        f_class.write(i+'\n')

    print("有"+str(blank_tag)+"个文书的行业标记为空！")
    print("train:")
    print(train_class_tag)
    print("val:")
    print(val_class_tag)
    print("test:")
    print(test_class_tag)
    f_train.close()
    f_test.close()
    # f_val.close()
    f_class.close()


# 这里试一下分层采样会不会好一些
# 将数据按照类别进行分层划分
def save_file_stratified(filename, ssdfile_dir, categories):
    """
    将文件分流到3个文件中
    filename: 原数据地址，一个csv文件
    文件内容格式:  类别\t内容
    """
    f_train = open('data/betadata2-711depart/train.txt', 'w', encoding='utf-8')
    f_val = open('data/betadata2-711depart/val.txt', 'w', encoding='utf-8')
    f_test = open('data/betadata2-711depart/test.txt', 'w', encoding='utf-8')

    # 事先将事实段全文和文件名之间建立一个字典
    dict_ssdqw = {}
    for ssdfile in os.listdir(ssdfile_dir):
        ssdfile_name = os.path.join(ssdfile_dir, ssdfile)
        f = open(ssdfile_name, 'r', encoding='utf-8')
        content_qw = ''
        content = f.readline()
        # 以下部分，因为统计整个案件基本情况他有换行，所以将多行处理在一行里面
        while content:
            content_qw += content
            content_qw = content_qw.replace('\n', '')
            content = f.readline()
        ssdfile_key = str(ssdfile).replace('.txt','')
        flag = True
        # **********************************************************************************************************
        # 本来是直接要建立字典，这里先做一个筛选，将小于50字的筛掉
        content_qw_list = list(content_qw)

        while ' ' in content_qw_list:
            content_qw_list.remove(' ')
        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                             '(', '／', '％', '…', '［', '］',
                             '$', '*', '【', '】', '？', '”', '，', '。', '”', '）', '（', '；', '《', '》', '-']
        for c in remove_characters:
            while c in content_qw_list:
                content_qw_list.remove(c)
        character_count = len(content_qw_list)
        index = int(character_count / 100)
        if index == 0:
            if len(content_qw_list) <= 50:
                flag = False
        # **********************************************************************************************************
        if flag:
            dict_ssdqw[ssdfile_key] = content_qw

    # doc_count代表每一类数据总共有多少个
    doc_count_0 = 0
    doc_count_1 = 0
    doc_count_2 = 0
    doc_count_3 = 0
    doc_count_4 = 0
    doc_count_5 = 0
    doc_count_6 = 0
    doc_count_7 = 0
    doc_count_8 = 0
    doc_count_9 = 0
    doc_count_10 = 0
    doc_count_11 = 0
    doc_count_12 = 0
    temp_file = open(filename, 'r', encoding='utf-8')
    line = temp_file.readline()
    while line:
        line_content = line.split(',')
        name = line_content[0]
        if name in dict_ssdqw:
            label = line_content[1]
            if label == categories[0]:
                doc_count_0 += 1
            elif label == categories[1]:
                doc_count_1 += 1
            elif label == categories[2]:
                doc_count_2 += 1
            elif label == categories[3]:
                doc_count_3 += 1
            elif label == categories[4]:
                doc_count_4 += 1
            elif label == categories[5]:
                doc_count_5 += 1
            elif label == categories[6]:
                doc_count_6 += 1
            elif label == categories[7]:
                doc_count_7 += 1
            elif label == categories[8]:
                doc_count_8 += 1
            elif label == categories[9]:
                doc_count_9 += 1
            elif label == categories[10]:
                doc_count_10 += 1
            elif label == categories[11]:
                doc_count_11 += 1
            elif label == categories[12]:
                doc_count_12 += 1
        line = temp_file.readline()
    temp_file.close()

    # 总数量
    doc_count = doc_count_0 + doc_count_1 + doc_count_2 + doc_count_3 +\
        doc_count_4 + doc_count_5 + doc_count_6 + doc_count_7 +\
        doc_count_8 + doc_count_9 + doc_count_10 + doc_count_11 + doc_count_12
    class_set = set()
    tag_train_0 = doc_count_0 * 70 / 100
    tag_train_1 = doc_count_1 * 70 / 100
    tag_train_2 = doc_count_2 * 70 / 100
    tag_train_3 = doc_count_3 * 70 / 100
    tag_train_4 = doc_count_4 * 70 / 100
    tag_train_5 = doc_count_5 * 70 / 100
    tag_train_6 = doc_count_6 * 70 / 100
    tag_train_7 = doc_count_7 * 70 / 100
    tag_train_8 = doc_count_8 * 70 / 100
    tag_train_9 = doc_count_9 * 70 / 100
    tag_train_10 = doc_count_10 * 70 / 100
    tag_train_11= doc_count_11 * 70 / 100
    tag_train_12 = doc_count_12 * 70 / 100
    tag_val_0 = doc_count_0 * 85 / 100
    tag_val_1 = doc_count_1 * 85 / 100
    tag_val_2 = doc_count_2 * 85 / 100
    tag_val_3 = doc_count_3 * 85 / 100
    tag_val_4 = doc_count_4 * 85 / 100
    tag_val_5 = doc_count_5 * 85 / 100
    tag_val_6 = doc_count_6 * 85 / 100
    tag_val_7 = doc_count_7 * 85 / 100
    tag_val_8 = doc_count_8 * 85 / 100
    tag_val_9 = doc_count_9 * 85 / 100
    tag_val_10 = doc_count_10 * 85 / 100
    tag_val_11 = doc_count_11 * 85 / 100
    tag_val_12 = doc_count_12 * 85 / 100

    # tag_test = doc_count * 70 / 100
    tag_0 = 0
    tag_1 = 0
    tag_2 = 0
    tag_3 = 0
    tag_4 = 0
    tag_5 = 0
    tag_6 = 0
    tag_7 = 0
    tag_8 = 0
    tag_9 = 0
    tag_10 = 0
    tag_11 = 0
    tag_12 = 0
    # 有些文书行业标记是空！！我想看看有多少条？
    blank_tag = 0
    # 标记一下，每个类别有多少个训练集、验证集、测试集？
    train_class_tag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    val_class_tag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_class_tag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # csvfile = open(filename, 'r', encoding='utf-8')
    txtfile = open(filename, 'r', encoding='utf-8')
    process_line = txtfile.readline()
    while process_line:
        line_content = process_line.split(',')
        name = line_content[0]
        if name in dict_ssdqw:
            content = dict_ssdqw[name]
            label = line_content[1]
            # if label != '' and label != '其他行业':
            if label != '':
                class_set.add(label)
                # 对每一类进行分层采样
                if label == categories[0]:
                    tag_0 += 1
                    if tag_0 < tag_train_0:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[0] += 1
                    elif tag_0 < tag_val_0:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[0] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[0] += 1

                elif label == categories[1]:
                    tag_1 += 1
                    if tag_1 < tag_train_1:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[1] += 1
                    elif tag_1 < tag_val_1:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[1] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[1] += 1

                elif label == categories[2]:
                    tag_2 += 1
                    if tag_2 < tag_train_2:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[2] += 1
                    elif tag_2 < tag_val_2:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[2] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[2] += 1

                elif label == categories[3]:
                    tag_3 += 1
                    if tag_3 < tag_train_3:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[3] += 1
                    elif tag_3 < tag_val_3:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[3] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[3] += 1

                elif label == categories[4]:
                    tag_4 += 1
                    if tag_4 < tag_train_4:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[4] += 1
                    elif tag_4 < tag_val_4:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[4] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[4] += 1

                elif label == categories[5]:
                    tag_5 += 1
                    if tag_5 < tag_train_5:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[5] += 1
                    elif tag_5 < tag_val_5:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[5] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[5] += 1

                elif label == categories[6]:
                    tag_6 += 1
                    if tag_6 < tag_train_6:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[6] += 1
                    elif tag_6 < tag_val_6:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[6] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[6] += 1

                elif label == categories[7]:
                    tag_7 += 1
                    if tag_7 < tag_train_7:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[7] += 1
                    elif tag_7 < tag_val_7:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[7] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[7] += 1

                elif label == categories[8]:
                    tag_8 += 1
                    if tag_8 < tag_train_8:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[8] += 1
                    elif tag_8 < tag_val_8:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[8] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[8] += 1

                elif label == categories[9]:
                    tag_9 += 1
                    if tag_9 < tag_train_9:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[9] += 1
                    elif tag_9 < tag_val_9:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[9] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[9] += 1

                elif label == categories[10]:
                    tag_10 += 1
                    if tag_10 < tag_train_10:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[10] += 1
                    elif tag_10 < tag_val_10:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[10] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[10] += 1

                elif label == categories[11]:
                    tag_11 += 1
                    if tag_11 < tag_train_11:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[11] += 1
                    elif tag_11 < tag_val_11:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[11] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[11] += 1

                elif label == categories[12]:
                    tag_12 += 1
                    if tag_12 < tag_train_12:
                        f_train.write(label + '\t' + content + '\n')
                        train_class_tag[12] += 1
                    elif tag_12 < tag_val_12:
                        f_val.write(label + '\t' + content + '\n')
                        val_class_tag[12] += 1
                    else:
                        f_test.write(label + '\t' + content + '\n')
                        test_class_tag[12] += 1
            else:
                blank_tag += 1

        process_line = txtfile.readline()

    txtfile.close()
    print("有" + str(blank_tag) + "个文书的行业标记为空！")
    print("train:")
    print(train_class_tag)
    train_tag_total =0
    for i_total in train_class_tag:
        train_tag_total += i_total
    train_class_tag_distribute = []
    for i in train_class_tag:
        train_class_tag_distribute.append((i / train_tag_total) * 100)
    print("分布:")
    print(train_class_tag_distribute)
    print("val:")
    print(val_class_tag)
    val_tag_total = 0
    for i_total in val_class_tag:
        val_tag_total += i_total
    val_class_tag_distribute = []
    for i in val_class_tag:
        val_class_tag_distribute.append((i / val_tag_total) * 100)
    print("分布:")
    print(val_class_tag_distribute)
    print("test:")
    print(test_class_tag)
    test_tag_total = 0
    for i_total in test_class_tag:
        test_tag_total += i_total
    test_class_tag_distribute = []
    for i in test_class_tag:
        test_class_tag_distribute.append((i / test_tag_total) * 100)
    print("分布:")
    print(test_class_tag_distribute)
    f_train.close()
    f_test.close()
    f_val.close()


# 此函数用于在ms-yg和xs-zkss的txt都有了的情况下，结合分类csv文件，将所有数据写入到一个txt中
def write_files_together(ms_file_path, ms_label, xs_file_path, xs_label ):
    f_save = open('data/jointdata/all.txt', 'w', encoding='utf-8')
    ms_csv = pd.read_csv(ms_label, sep=',', encoding='GBK')
    ms_csv_value = ms_csv.values
    xs_csv = pd.read_csv(xs_label, sep=',', encoding='GBK')
    xs_csv_value = xs_csv.values

    cnt_ms = 0
    cnt_xs = 0
    # 这里还要统计是不是存在数据重复的情况
    dict_ssd_ah = {}
    dict_ah_hy = {}
    for data in ms_csv_value:
        # data[0] 案号   data[1] 行业  data[2] 全文
        ah_value = data[0]
        ah = ah_value + '.txt'
        hy = data[1]
        f_content_path = os.path.join(ms_file_path,ah)
        try:
            f_content = open(f_content_path,'r',encoding='utf-8')
            content = f_content.readline()
            content = str(content).replace('&ldquo；', '“')
            content = content.replace('&rdquo；', '”')
            f_content.close()
            # f_save.write(hy+'\t'+content+'\n')
            dict_ssd_ah[content] = ah_value
            dict_ah_hy[ah_value] = hy
            cnt_ms += 1
            print(str(cnt_ms))
        except FileNotFoundError:
            continue
    print(len(dict_ssd_ah))
    result_ms = sorted(dict_ssd_ah.items(),key=lambda item: item[0])
    for data in result_ms:
        if '查明' not in data[0]:
            continue
        if len(data[0]) < 150:
            continue
        print(len(data[0]),end=' ')
        print(data[0])
        f_save.write(dict_ah_hy[data[1]]+'\t'+data[0]+'\n')
    # for data in xs_csv_value:
    #     ah = data[0] + '.txt'
    #     hy = data[1]
    #     f_content_path = os.path.join(xs_file_path, ah)
    #     try:
    #         f_content = open(f_content_path, 'r', encoding='utf-8')
    #         content = f_content.readline()
    #         f_content.close()
    #         f_save.write(hy + '\t' + content + '\n')
    #
    #         cnt_xs += 1
    #         print('刑事：' + str(cnt_xs))
    #     except Exception:
    #         continue
    f_save.close()

    print('民事数据量：'+str(cnt_ms))
    print('刑事数据量：' + str(cnt_xs))


# 将数据按照编码转化后存入文件中
def data_convert():
    x_c, y = data_loader.process_file(all_data_dir, character_to_id, cat_to_id, config.seq_length_c)
    x_w, _ = data_loader_wordlevel.process_file(all_data_dir, word_to_id, cat_to_id, config.seq_length_w)
    file_x_c_id = open('data\\10-fold-original-data\\data-convert\\x_c_id.txt','w',encoding='utf-8')
    file_x_w_id = open('data\\10-fold-original-data\\data-convert\\x_w_id.txt','w',encoding='utf-8')
    file_y_id = open('data\\10-fold-original-data\\data-convert\\y_id.txt','w',encoding='utf-8')
    for data in x_c:
        print(len(data))
        for i in data:
            file_x_c_id.write(str(i) + ' ')
        file_x_c_id.write('\n')
    for data in x_w:
        print(len(data))
        for i in data:
            file_x_w_id.write(str(i) + ' ')
        file_x_w_id.write('\n')
    for data in y:
        print(len(data))
        for i in data:
            file_y_id.write(str(i) + ' ')
        file_y_id.write('\n')

    file_x_c_id.close()
    file_x_w_id.close()
    file_y_id.close()


def stratified_cross(base_path, x_c_file_path, x_w_file_path, y_file_path):
    x_c_file = open(x_c_file_path,'r',encoding='utf-8')
    x_w_file = open(x_w_file_path, 'r', encoding='utf-8')
    y_file = open(y_file_path,'r',encoding='utf-8')
    x_c_list = []
    x_w_list = []
    y_list = []

    x_c_line = x_c_file.readline()
    while x_c_line:
        content_list = [int(x) for x in x_c_line.strip().split()]
        x_c_list.append(content_list)
        x_c_line = x_c_file.readline()

    x_w_line = x_w_file.readline()
    while x_w_line:
        content_list = [int(x) for x in x_w_line.strip().split()]
        x_w_list.append(content_list)
        x_w_line = x_w_file.readline()

    y_line = y_file.readline()
    while y_line:
        content_list = [float(y) for y in y_line.strip().split()]
        y_list.append(content_list)
        y_line = y_file.readline()

    x_c_ndarray = np.array(x_c_list)
    x_w_ndarray = np.array(x_w_list)
    y_ndarray = np.array(y_list)
    y_ndarray_not_onehot = [np.argmax(i) for i in y_ndarray]

    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
    folder_index = 0
    for train, test in sfolder.split(x_c_ndarray, y_ndarray_not_onehot):
        folder_index += 1
        x_c_train_file_name = os.path.join(base_path, str('train_c_' + str(folder_index) + '.txt'))
        x_c_train_file = open(x_c_train_file_name,'w',encoding='utf-8')
        x_w_train_file_name = os.path.join(base_path, str('train_w_' + str(folder_index) + '.txt'))
        x_w_train_file = open(x_w_train_file_name, 'w', encoding='utf-8')
        y_train_file_name = os.path.join(base_path, str('train_y_' + str(folder_index) + '.txt'))
        y_train_file = open(y_train_file_name, 'w', encoding='utf-8')

        x_c_test_file_name = os.path.join(base_path, str('test_c_' + str(folder_index) + '.txt'))
        x_c_test_file = open(x_c_test_file_name,'w',encoding='utf-8')
        x_w_test_file_name = os.path.join(base_path, str('test_w_' + str(folder_index) + '.txt'))
        x_w_test_file = open(x_w_test_file_name, 'w', encoding='utf-8')
        y_test_file_name = os.path.join(base_path, str('test_y_' + str(folder_index) + '.txt'))
        y_test_file = open(y_test_file_name, 'w', encoding='utf-8')

        print('Train: %s | test: %s' % (train, test))
        x_c_train = x_c_ndarray[train]
        x_w_train = x_w_ndarray[train]
        y_train = y_ndarray[train]

        x_c_test = x_c_ndarray[test]
        x_w_test = x_w_ndarray[test]
        y_test = y_ndarray[test]

        for data in x_c_train:
            for i_data in data:
                x_c_train_file.write(str(i_data) + ' ')
            x_c_train_file.write('\n')

        for data in x_w_train:
            for i_data in data:
                x_w_train_file.write(str(i_data) + ' ')
            x_w_train_file.write('\n')

        for data in y_train:
            for i_data in data:
                y_train_file.write(str(i_data) + ' ')
            y_train_file.write('\n')

        for data in x_c_test:
            for i_data in data:
                x_c_test_file.write(str(i_data) + ' ')
            x_c_test_file.write('\n')

        for data in x_w_test:
            for i_data in data:
                x_w_test_file.write(str(i_data) + ' ')
            x_w_test_file.write('\n')

        for data in y_test:
            for i_data in data:
                y_test_file.write(str(i_data) + ' ')
            y_test_file.write('\n')

        x_c_train_file.close()
        x_w_train_file.close()
        y_train_file.close()
        x_c_test_file.close()
        x_w_test_file.close()
        y_test_file.close()


# 将CNews子数据集整理正一个allnews.txt
def group_data_thucnews(file_path,save_dir):
    cnt_all = 0
    save_file = open(save_dir,'w',encoding='utf-8')
    for category in os.listdir(file_path):
        # print(category)
        label = category
        for txt_file in os.listdir(os.path.join(file_path,label)):
            file = open(os.path.join(file_path,label,txt_file),'r',encoding='utf-8')
            content_all = ''
            content = file.readline()
            while content:
                content = content.strip()
                while ' ' in content:
                    content = content.replace(' ','')
                while '\t' in content:
                    content = content.replace('\t','')
                while '\n' in content:
                    content = content.replace('\n','。')
                content_all += content
                content = file.readline()
            # print(content_all)
            save_file.write(label+'\t'+content_all+'\n')
            cnt_all += 1
            print(cnt_all)


def split_all_data():
    # file = open('data/cnews/all.txt','r',encoding='utf-8')
    # train_file = open('data/cnews/train.txt','w',encoding='utf-8')
    # val_file = open('data/cnews/val.txt','w',encoding='utf-8')
    # test_file = open('data/cnews/test.txt','w',encoding='utf-8')

    file = open('E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\group-data\\out.txt', 'r', encoding='utf-8')
    train_file = open('E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\group-data\\train.txt', 'w', encoding='utf-8')
    val_file = open('E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\group-data\\val.txt', 'w', encoding='utf-8')
    test_file = open('E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\group-data\\test.txt', 'w', encoding='utf-8')

    cnt = len(file.readlines())
    file.close()
    file = open('E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\group-data\\out.txt', 'r', encoding='utf-8')
    train_cnt = int(cnt * 0.7)
    val_cnt = int(cnt * 0.85)
    line = file.readline()
    index = 0
    while line:
        index += 1
        print(index)
        split_list = line.strip().split('\t')
        if len(split_list) < 2:
            line = file.readline()
            continue
        label = split_list[0]
        content = split_list[1]

        if index < train_cnt:
            train_file.write(label+'\t'+content+'\n')
        elif index < val_cnt:
            val_file.write(label+'\t'+content+'\n')
        else:
            test_file.write(label+'\t'+content+'\n')

        line = file.readline()
    train_file.close()
    val_file.close()
    test_file.close()


def group_imdb():
    file_pos_path = 'E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\train\\pos'
    file_neg_path = 'E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\train\\neg'
    file_group = 'E:\\Master Related\\Public Dataset\\aclImdb_v1\\aclImdb\\group-data\\all.txt'
    file_write = open(file_group,'w',encoding='utf-8')

    for txt_file in os.listdir(file_pos_path):
        file = open(os.path.join(file_pos_path, txt_file), 'r', encoding='utf-8')
        content = file.readline()
        file.close()
        file_write.write('pos'+'\t'+content+'\n')

    for txt_file in os.listdir(file_neg_path):
        file = open(os.path.join(file_neg_path, txt_file), 'r', encoding='utf-8')
        content = file.readline()
        file.close()
        file_write.write('neg'+'\t'+content+'\n')

    file_write.close()


if __name__ == '__main__':
    categories = [
        "机械制造行业",
        "五金建材行业",
        "农林牧渔行业",
        "化工行业",
        "电子通讯行业",
        "文体生活用品行业",
        "农副食品行业",
        "纺织服饰行业",
        "家电行业",
        "其他行业",
        "食品药品行业",
        "交通运输行业",
        "酒水饮料茶奶行业"
    ]
    # save_file('../data/qwdata/shuffle-try2/classified_table_ms.txt', '../data/qwdata/ms-ygscplusssdqw')
    # save_file_stratified('data/qwdata/shuffle-try3/classified_table_ms.txt', 'data/qwdata/ms-ygscplusssdqw-clean',categories)
    # print(len(open('data/betadata2-711depart/train.txt', 'r', encoding='utf-8').readlines()))
    # print(len(open('data/betadata2-711depart/val.txt', 'r', encoding='utf-8').readlines()))
    # print(len(open('data/betadata2-711depart/test.txt', 'r', encoding='utf-8').readlines()))
    # write_files_together(ms_file_path='data\\qwdata\\ms-ssd',
    #                      ms_label='data\\qwdata\\classifier_ms.csv',
    #                      xs_file_path='data\\qwdata\\new-xs-zkss-ajjbkq',
    #                      xs_label='data\\qwdata\\classifier_xs.csv')



    config = TCNNConfig()
    config.vocab_size_w = 200000
    config.vocab_size_c = 200000
    if not os.path.exists(vocab_dir_c):  # 如果不存在字表，重建
        data_loader.build_vocab(all_data_dir, vocab_dir_c, config.vocab_size_c)
    if not os.path.exists(vocab_dir_w):  # 如果不存在词汇表，重建
        data_loader_wordlevel.build_vocab(all_data_dir, vocab_dir_w, config.vocab_size_w)


    # categories, cat_to_id = data_loader.read_category()
    # characters, character_to_id = data_loader.read_vocab(vocab_dir_c)
    # words, word_to_id = data_loader_wordlevel.read_vocab(vocab_dir_w)
    # config.vocab_size_c = len(characters)
    # config.vocab_size_w = len(words)
    #
    # config.seq_length_c = 1500
    # config.seq_length_w = 800

    # data_convert()
    # stratified_cross('data\\10-fold-original-data\\data-convert',
    #                  'data\\10-fold-original-data\\data-convert\\x_c_id.txt',
    #                  'data\\10-fold-original-data\\data-convert\\x_w_id.txt',
    #                  'data\\10-fold-original-data\\data-convert\\y_id.txt')

    # group_data_thucnews('E:\\Master Related\\Public Dataset\\SubTHUCNews','data/cnews/all.txt')
    # split_all_data()

    # group_imdb()
    # split_all_data()
