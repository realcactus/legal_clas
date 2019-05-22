# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     check_ssd_distribution
   Author :        Xiaosong Zhou
   date：          2019/5/8
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'
import os
import re

# 这个文件的作用是，现在有了一些文书全文的txt文件，但是是全文，希望从全文中提取出事实段部分，但是是自己写正则，所以需要检查一下事实段分布


def check_re_distribution(file_path):
    path_list = os.listdir(file_path)
    cnt = 0
    cnt_ssss = 0
    cnt_slcm = 0
    for file_name in path_list:
        cnt += 1
        print(cnt)
        file = os.path.join(file_path,file_name)
        f = open(file,encoding='utf-8')
        content = f.readline()
        # if '事实' in content:
        #     cnt_ssss += 1
        if re.match(r'.*审理.*查明.*',content) and '上述事实' in content:
            cnt_slcm+=1
        f.close()

    print('总文书: ' + str(cnt))
    print('含有上述事实: ' + str(cnt_ssss))
    print('含有*审理.*查明.*' + str(cnt_slcm))


def check_stratified_distribution(X, Y):
    # 检查X的label分布是否分层
    return 0


if __name__ == '__main__':
    check_re_distribution(file_path='D:/MYSQL/backup/xs_txt')