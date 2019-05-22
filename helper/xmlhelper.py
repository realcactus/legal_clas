#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'zhouxiaosong'

import xml.dom.minidom
import os


def xml2shortdoc(filedir):
    save_dir = 'data/qwdata/ms-ssd'

    wrong_tag = 0
    wrong_tag_at_all = 0
    for xmlfile in os.listdir(filedir):
        filename = os.path.join(filedir, xmlfile)
        dom = xml.dom.minidom.parse(filename)
        root = dom.documentElement
        ajjbqk_node = root.getElementsByTagName("AJJBQK")
        if len(ajjbqk_node) == 0:
            wrong_tag_at_all += 1
            print('####################################################################')
            print(xmlfile)
            print('####################################################################')
            continue
        ajjbqk = ajjbqk_node[0]
        ajjbqk_qw = ajjbqk.getAttribute("value")
        cmssd_node = ajjbqk.getElementsByTagName("CMSSD")
        ssd_qw = ''
        if len(cmssd_node)>0:
            cmssd = ajjbqk.getElementsByTagName("CMSSD")[0]
            # 对此次分类有用的事实段内容
            ssd_qw = cmssd.getAttribute("value")
        else:
            wrong_tag += 1
            print(xmlfile)
            ssd_qw = ajjbqk_qw

        qw = ssd_qw
        # 保存路径
        save_filename = os.path.join(save_dir, xmlfile)
        # 改成txt保存
        save_filename = str(save_filename).replace('xml', 'txt')
        f_save = open(save_filename, 'w', encoding='utf-8')
        f_save.write(qw)
        f_save.close()

    print(wrong_tag_at_all)
    print(wrong_tag)


def xs_xml2shortdoc(filedir):
    save_dir = 'data/qwdata/xs-ssd'

    wrong_tag = 0
    wrong_tag_at_all = 0
    for xmlfile in os.listdir(filedir):
        filename = os.path.join(filedir, xmlfile)
        dom = xml.dom.minidom.parse(filename)
        root = dom.documentElement
        ajjbqk_node = root.getElementsByTagName("AJJBQK")
        if len(ajjbqk_node) == 0:
            wrong_tag_at_all += 1
            print('####################################################################')
            print(xmlfile)
            print('####################################################################')
            continue
        ajjbqk = ajjbqk_node[0]
        ajjbqk_qw = ajjbqk.getAttribute("value")
        zkdl_node = ajjbqk.getElementsByTagName('ZKDL')

        ssd_qw = ''
        if len(zkdl_node)>0:
            zkdl = zkdl_node[0]
            zkss_node = zkdl.getElementsByTagName('ZKSS')
            if len(zkss_node) > 0:
                zkss = zkss_node[0]
                # 对此次分类有用的事实段内容
                ssd_qw = zkss.getAttribute("value")
        else:
            wrong_tag += 1
            print(xmlfile)
            ssd_qw = ajjbqk_qw

        qw = ssd_qw
        # 保存路径
        save_filename = os.path.join(save_dir, xmlfile)
        # 改成txt保存
        save_filename = str(save_filename).replace('xml', 'txt')
        f_save = open(save_filename, 'w', encoding='utf-8')
        f_save.write(qw)
        f_save.close()

    print(wrong_tag_at_all)
    print(wrong_tag)


if __name__ == '__main__':
    xml2shortdoc('D:\\MYSQL\\backup\\ms_xml_clear')