# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cal_distribution
   Description :
   Author :       xszhou
   date：          2019/1/22
-------------------------------------------------
"""
__author__ = 'xszhou'


# 计算数据字数的分布

# 100字为单位
# 0-100  101-200  201-300 .....

# i从 0到99
# 代表字数为i*100 <= x < i+1 *100 的文档有多少篇
distribution = [0 for i in range(91)]
category_distribution = [0 for i in range(13)]

import os
import jieba
import numpy as np

def cal_dis(path_dir):
    for ssdfile in os.listdir(path_dir):
        ssdfile_name = os.path.join(path_dir, ssdfile)
        f = open(ssdfile_name, 'r', encoding='utf-8')
        content_qw = ''
        content = f.readline()
        # 以下部分，因为统计整个案件基本情况他有换行，所以将多行处理在一行里面
        while content:
            content_qw += content
            content_qw = content_qw.replace('\n', '')
            content = f.readline()
        # 这里统计字数，考虑到和后面处理数据一致，这里也做一些处理
        content_qw = list(content_qw)

        while ' ' in content_qw:
            content_qw.remove(' ')
        remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                             '(', '／', '％', '…', '［', '］',
                             '$', '*', '【', '】', '？', '”', '，', '。', '”', '）', '（', '；', '《', '》', '-']
        for c in remove_characters:
            while c in content_qw:
                content_qw.remove(c)
        character_count = len(content_qw)
        index = int(character_count / 100)
        if index == 1:
            if len(content_qw) <= 500:
                for i in content_qw:
                    print(i, end='')
                print('||||||||||||||||||||||||||', end='')
                print(ssdfile_name)
        distribution[index] += 1


def cal_dis_from_txt(path_dir):
    txt_file = open(path_dir,'r',encoding='utf-8')
    txt = txt_file.readline()
    # max_count = 0
    while txt:
        label, content = txt.strip().split('\t')
        content_qw = list(content)
        count = len(content_qw)
        # max_count = max(max_count, count)
        index = int(count / 100)
        distribution[index] += 1

        txt = txt_file.readline()
    # 15936
    # print(str(max_count))
    print(distribution)
    total_data = 0
    index = -1
    for data in distribution:
        index += 1
        if index > 30:
            break
        total_data += data
        print(str(index) + ', '+str(data))
    print(total_data)


def cal_dis_from_txt_word_level(path_dir):
    txt_file = open(path_dir,'r',encoding='utf-8')
    txt = txt_file.readline()
    # max_count = 0
    while txt:
        label, content = txt.strip().split('\t')
        result = jieba.cut(content)
        # contents.append(list(str(content).lower()))
        temp_list = list(result)
        count = len(temp_list)
        # max_count = max(max_count, count)
        index = int(count / 100)
        distribution[index] += 1

        txt = txt_file.readline()
    # 9051
    # print(str(max_count))
    print(distribution)
    total_data = 0
    index = -1
    for data in distribution:
        index += 1
        if index > 17:
            break
        total_data += data
        print(str(index) + ', '+str(data))
    print(total_data)


def cal_category_distribution(file_path):
    file = open(file_path,'r',encoding='utf-8')
    line = file.readline()
    while line:
        content_list = [float(x) for x in line.strip().split()]
        category_ndarray = np.array(content_list)
        index = -1
        for i in category_ndarray:
            index += 1
            if i == 1:
                category_distribution[index] += 1
        line = file.readline()


if __name__ =='__main__':
    # cal_dis_from_txt_word_level('data/10-fold-cross-validation-data/all.txt')
    # 未经预处理，字符序列3000个字符以内的有5597条，总共5840条，占比95.84%
    # 未经预处理，词语序列1700个词以内的有5603条，总共5840条，占比95.94%

    # 所有训练数据（共5837条）分布[366, 552, 576, 1376, 68, 258, 214, 270, 166, 88, 1274, 432, 197]
    # cal_category_distribution('data/10-fold-cross-validation-data/data-convert/y_id.txt')
    print(category_distribution)
    print('hello')

