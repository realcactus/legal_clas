# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     csv2txt
   Author :        Xiaosong Zhou
   date：          2019/5/6
-------------------------------------------------
"""
import os,shutil

__author__ = 'Xiaosong Zhou'
import pandas as pd

# 这部分的功能是将csv文件中的全文部分按照名字写入xml文件
# 首先是要写入txt文件，然后用之前实验室工作的txt2xml部分代码转化成xml
# 这个文件主要是写入txt的，这里这样命名是因为要检查一下xml文件


def write_csv_to_txt(file_path,save_path):
    xs_csv = pd.read_csv(file_path, sep=',', encoding='gbk')
    xs_csv_value = xs_csv.values
    for data in xs_csv_value:
        name = data[0]
        content = data[2]
        file_name = save_path + '/' + name + '.txt'
        try:
            file_out = open(file_name,'w',encoding='utf-8')
            file_out.write(content)
            file_out.close()
        except IOError:
            print(file_name)
            continue


def write_txt_with_enter(file_path,save_path):
    # 将原始一行的txt转化为包含回车的格式
    path_list = os.listdir(file_path)
    for file_name in path_list:
        file = os.path.join(file_path,file_name)
        f = open(file,'r',encoding='utf-8')
        content = f.readline()
        file_save_name = os.path.join(save_path,file_name)
        f_w = open(file_save_name, 'w', encoding='utf-8')
        contents = content.split()
        for i_content in contents:
            f_w.write(i_content+'\n')
        f_w.close()
        f.close()


# 检查后缀名是.txt.xml还是.txt（.txt是没有成功转换的）
def check_xml(file_path, save_path):
    path_list = os.listdir(file_path)
    for file_name in path_list:
        file_type = os.path.splitext(file_name)
        if file_type[1] == '.xml':
            source_file_path = os.path.join(file_path, file_name)
            save_file_name = file_type[0].split('.')[0]
            save_file_name += '.xml'
            save_file_path = os.path.join(save_path,save_file_name)
            shutil.copyfile(source_file_path,save_file_path)


if __name__ == '__main__':
    # write_csv_to_txt(file_path='D:/MYSQL/backup/classifier_xs.csv', save_path='D:/MYSQL/backup/xs_txt')
    # write_csv_to_txt(file_path='D:/MYSQL/backup/classifier_ms.csv', save_path='D:/MYSQL/backup/
    # write_txt_with_enter(file_path='D:/MYSQL/backup/xs_txt',save_path='D:/MYSQL/backup/xs_txt_enter')
    check_xml(file_path='D:/MYSQL/backup/xs_xml',save_path='D:/MYSQL/backup/xs_xml_clear')