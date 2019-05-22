#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'zhouxiaosong'


import os
import csv
import codecs


def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')


def save_file(filename):
    dirname = '../data/qwdata/ms'
    filename_set = '../data/qwdata/ms.csv'
    ah_set = set()
    csvfile_xs_set = open(filename_set, 'r', encoding='utf-8-sig')

    for line in csv.reader(csvfile_xs_set, delimiter=',', quotechar='"'):
        ah_set.add(line[1])
    csvfile_xs_set.close()

    csvfile_xs = open(filename, 'r', encoding='utf-8-sig')
    true_tag = 0
    tag = 0
    for line in csv.reader(csvfile_xs, delimiter=',', quotechar='"'):
        tag += 1
        ah = line[0]
        content = line[4]
        content = str(content).replace('&ldquo；', '“')
        content = content.replace('&rdquo；', '”')
        content = content.replace(' ', '\n')
        if ah in ah_set:
            ah = ah + '.txt'
            write_file = os.path.join(dirname, ah)
            f = open(write_file, 'w', encoding='utf-8')
            f.write(content)
            true_tag += 1
            f.close()
    print(true_tag)



if __name__ == '__main__':
    save_file('../data/qwdata/classified_table_ms.csv')
    # print(len(open('../data/usefuldata/train.txt', 'r', encoding='utf-8').readlines()))
    # print(len(open('../data/usefuldata/val.txt', 'r', encoding='utf-8').readlines()))
    # print(len(open('../data/usefuldata/test.txt', 'r', encoding='utf-8').readlines()))
    # print(len(open('../data/usefuldata/class.txt', 'r', encoding='utf-8').readlines()))
