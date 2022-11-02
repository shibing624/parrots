#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
语音识别的语言模型

基于马尔可夫模型的语言模型
"""
import os
from loguru import logger

pwd_path = os.path.abspath(os.path.dirname(__file__))
pinyin2hanzi_dir = os.path.join(pwd_path, 'data/pinyin2hanzi')


class Pinyin2Hanzi(object):
    def __init__(self, model_dir=pinyin2hanzi_dir):
        self.dict_pinyin = self.get_symbol_dict(os.path.join(model_dir, 'pinyin_hanzi_dict.txt'))
        self.model1 = self.get_model_file(os.path.join(model_dir, 'char_idx.txt'))
        self.model2 = self.get_model_file(os.path.join(model_dir, 'word_idx.txt'))
        self.pinyin = self.get_pinyin(os.path.join(model_dir, 'dic_pinyin.txt'))
        self.model = (self.dict_pinyin, self.model1, self.model2)

    def pinyin_2_hanzi(self, list_syllable):
        """
        语音拼音 => 文本
        :param list_syllable:
        :return:
        """
        r = ''
        length = len(list_syllable)
        if not length:  # 传入的参数没有包含任何拼音时
            return ''

        # 先取出一个字，即拼音列表中第一个字
        str_tmp = [list_syllable[0]]

        for i in range(0, length - 1):
            # 依次从第一个字开始每次连续取两个字拼音
            str_split = list_syllable[i] + ' ' + list_syllable[i + 1]
            # print(str_split,str_tmp,r)
            # 如果这个拼音在汉语拼音状态转移字典里的话
            if str_split in self.pinyin:
                # 将第二个字的拼音加入
                str_tmp.append(list_syllable[i + 1])
            else:
                # 否则不加入，然后直接将现有的拼音序列进行解码
                str_decode = self.decode(str_tmp, 0.0000)
                # print('decode ',str_tmp,str_decode)
                if (str_decode != []):
                    r += str_decode[0][0]
                # 再重新从i+1开始作为第一个拼音
                str_tmp = [list_syllable[i + 1]]
        str_decode = self.decode(str_tmp, 0.0000)
        if str_decode:
            r += str_decode[0][0]
        return r

    def decode(self, list_syllable, yuzhi=0.0001):
        """
        实现拼音向文本的转换
        基于马尔可夫链
        """
        list_words = []
        num_pinyin = len(list_syllable)
        # 开始语音解码
        for i in range(num_pinyin):
            ls = ''
            if list_syllable[i] in self.dict_pinyin:  # 如果这个拼音在汉语拼音字典里的话
                # 获取拼音下属的字的列表，ls包含了该拼音对应的所有的字
                ls = self.dict_pinyin[list_syllable[i]]
            else:
                break

            if i == 0:
                # 第一个字做初始处理
                num_ls = len(ls)
                for j in range(num_ls):
                    tuple_word = ['', 0.0]
                    # 设置马尔科夫模型初始状态值
                    # 设置初始概率，置为1.0
                    tuple_word = [ls[j], 1.0]
                    # print(tuple_word)
                    # 添加到可能的句子列表
                    list_words.append(tuple_word)

                # print(list_words)
                continue
            else:
                # 开始处理紧跟在第一个字后面的字
                list_words_2 = []
                num_ls_word = len(list_words)
                # print('ls_wd: ',list_words)
                for j in range(0, num_ls_word):

                    num_ls = len(ls)
                    for k in range(0, num_ls):
                        tuple_word = ['', 0.0]
                        tuple_word = list(list_words[j])  # 把现有的每一条短语取出来
                        # print('tw1: ',tuple_word)
                        tuple_word[0] = tuple_word[0] + ls[k]  # 尝试按照下一个音可能对应的全部的字进行组合
                        # print('ls[k]  ',ls[k])

                        tmp_words = tuple_word[0][-2:]  # 取出用于计算的最后两个字
                        # print('tmp_words: ',tmp_words,tmp_words in self.model2)
                        if tmp_words in self.model2:  # 判断它们是不是再状态转移表里
                            # print(tmp_words,tmp_words in self.model2)
                            tuple_word[1] = tuple_word[1] * float(self.model2[tmp_words]) / float(
                                self.model1[tmp_words[-2]])
                        # 核心！在当前概率上乘转移概率，公式化简后为第n-1和n个字出现的次数除以第n-1个字出现的次数
                        # print(self.model2[tmp_words],self.model1[tmp_words[-2]])
                        else:
                            tuple_word[1] = 0.0
                            continue
                        # print('tw2: ',tuple_word)
                        # print(tuple_word[1] >= pow(yuzhi, i))
                        if tuple_word[1] >= pow(yuzhi, i):
                            # 大于阈值之后保留，否则丢弃
                            list_words_2.append(tuple_word)

                list_words = list_words_2
                # print(list_words,'\n')
        # print(list_words)
        for i in range(0, len(list_words)):
            for j in range(i + 1, len(list_words)):
                if (list_words[i][1] < list_words[j][1]):
                    tmp = list_words[i]
                    list_words[i] = list_words[j]
                    list_words[j] = tmp

        return list_words

    def get_symbol_dict(self, file_path):
        """
        读取拼音汉字的字典文件
        :param file_path:
        :return: 读取后的字典
        """
        txt_obj = open(file_path, 'r', encoding='utf-8')  # 打开文件并读入
        txt_text = txt_obj.read()
        txt_obj.close()
        txt_lines = txt_text.split('\n')  # 文本分割

        dic_symbol = {}  # 初始化符号字典
        for i in txt_lines:
            list_symbol = []  # 初始化符号列表
            if i:
                txt_l = i.split('\t')
                pinyin = txt_l[0]
                for word in txt_l[1]:
                    list_symbol.append(word)
            dic_symbol[pinyin] = list_symbol
        logger.debug('Loaded: %s, size: %d' % (file_path, len(dic_symbol)))
        return dic_symbol

    def get_model_file(self, model_path):
        """
        读取语言模型的文件
        :param model_path:
        :return: 读取后的模型
        """
        txt_obj = open(model_path, 'r', encoding='utf-8')  # 打开文件并读入
        txt_text = txt_obj.read()
        txt_obj.close()
        txt_lines = txt_text.split('\n')  # 文本分割

        dic_model = {}  # 初始化符号字典
        for i in txt_lines:
            if i:
                txt_l = i.split('\t')
                if (len(txt_l) == 1):
                    continue
                dic_model[txt_l[0]] = txt_l[1]
        logger.debug('Loaded: %s, size: %d' % (model_path, len(dic_model)))
        return dic_model

    def get_pinyin(self, filename):
        file_obj = open(filename, 'r', encoding='utf-8')
        txt_all = file_obj.read()
        file_obj.close()

        txt_lines = txt_all.split('\n')
        dic = {}
        for line in txt_lines:
            if not line:
                continue
            pinyin_split = line.split('\t')
            list_pinyin = pinyin_split[0]
            if (list_pinyin not in dic) and int(pinyin_split[1]) > 1:
                dic[list_pinyin] = pinyin_split[1]
        logger.debug('Loaded: %s, size: %d' % (filename, len(dic)))
        return dic
