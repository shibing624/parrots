# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: math
"""

import difflib


def edit_distance(str1, str2):
    distance = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        # print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            distance += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            distance += (j2 - j1)
        elif tag == 'delete':
            distance += (i2 - i1)
    return distance
