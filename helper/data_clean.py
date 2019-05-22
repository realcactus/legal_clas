# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_clean
   Description :
   Author :       xszhou
   date：          2019/1/22
-------------------------------------------------
"""
__author__ = 'xszhou'


# 事先提取查明事实段全文

# ms-ssd文件夹里面7965条txt文件

import os
import re

def clean_data(file_path):
    # 查重

    # 事先将事实段全文和文件名之间建立一个字典
    dict_ssdqw = {}
    for ssdfile in os.listdir(file_path):
        ssdfile_name = os.path.join(file_path, ssdfile)
        f = open(ssdfile_name, 'r', encoding='utf-8')
        content_qw = ''
        content = f.readline()
        # 以下部分，因为统计整个案件基本情况他有换行，所以将多行处理在一行里面
        while content:
            content_qw += content
            content_qw = content_qw.replace('\n', '')
            content = f.readline()
        f.close()
        # *******************************************************************
        # 处理清洗过程
        content_qw = content_qw.replace('&times；','')
        content_qw = content_qw.replace('&ldquo；', '“')
        content_qw = content_qw.replace('&rdquo；', '”')

        str = content_qw
        str = re.sub(r'×+', '', str)
        str = re.sub(r' +', '', str)

        # 去除原告诉称语句模式
        str = re.sub(r'^.*诉称','',str)

        # 过滤车牌，一共34种车牌
        str = re.sub(r'内蒙古[0-9A-Za-z]+[牌]*', '', str)
        str = re.sub(r'[京津冀晋辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新港澳台][0-9A-Za-z]+[牌]*', '', str)
        # 过滤时间
        str = re.sub(r'[0-9]+年', '', str)
        str = re.sub(r'[0-9]+月', '', str)
        str = re.sub(r'[0-9]+日', '', str)
        str = re.sub(r'[0-9]+年[0-9]+月', '', str)
        str = re.sub(r'[0-9]+月[0-9]+日', '', str)
        str = re.sub(r'[0-9]+年[0-9]+月[0-9]+日', '', str)

        str = re.sub(r'[A-Za-z]+[0-9]+', '', str)
        # ****************************************
        # 从法律停用词中去除一些词语
        keyword_filter_file = open('helper/legal_stop.txt', 'r', encoding='utf-8')
        keyword_filter = []
        index_tmp = 0
        for line in keyword_filter_file:
            index_tmp += 1
            keyword_filter.append(line.strip())
        print(index_tmp)
        keyword_filter_file.close()
        # ****************************************

        for legal_word in keyword_filter:
            str = re.sub(legal_word, '', str)

        print(str)
        # *******************************************************************

        f_write = open(ssdfile_name, 'w', encoding='utf-8')
        f_write.write(str)
        f_write.close()

if __name__=='__main__':
    clean_data('data/qwdata/ms-ygscplusssdqw')
