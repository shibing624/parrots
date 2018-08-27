# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 获取符号字典列表的程序
"""


def get_pinyin_list(dict_path=''):
    """
    加载拼音符号列表，用于标记符号
    :param dict_path: 拼音符号列表
    :return:
    """
    list_symbol = []  # 符号列表
    pinyin_idx = 0
    with open(dict_path, mode='r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip('\n')
            parts = line.split('\t')
            list_symbol.append(parts[pinyin_idx])
    list_symbol.append('_')
    return list_symbol
