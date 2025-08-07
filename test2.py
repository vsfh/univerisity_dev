import os
import numpy as np

# 2 string pure number 

def add_num(str1, str2):
    add_one = 0
    res = ''

    length = min(len(str1), len(str2))
    max_len = max(len(str1), len(str2))
    for i in range(length):
        index = i + 1
        cur = int(str1[-index]) + int(str2[-index]) + add_one
        add_one = cur // 10
        cur = cur % 10
        res = res + str(cur)

    if len(str1) > len(str2):
        cur_str = str1
    else:
        cur_str = str2
    for i in range(length, max_len):
        index = i + 1
        cur = int(cur_str[-index]) + add_one
        add_one = cur // 10
        cur = cur % 10
        res = res + str(cur)
    if add_one > 0:
        res = res + '1' 
    return res[::-1]

def minus_num():
    pass

def add_num_one_minus(str1, str2):
    a =  len(str2) + 1 - len(str1)
    if a == 0:
        a = 1
        for i in range(len(str1)-1):
            if str1[i+1] > str2[i]:
                a = -1

    if a > 0:
        return minus_num(str2, str1)
    else:
        return minus_num(str1, str2)
            


def add_num_minus(str1, str2):
    if '-' in str1 and '-' in str2:
        return '-'+add_num(str1[1:], str2[1:])
    elif '-' in str1 and not '-' in str2:
        add_num_one_minus(str1, str2)
    elif '-' in str2 and not '-' in str1:
        add_num_one_minus(str2, str1)
    else:
        return add_num(str1, str2)
    
if __name__ == '__main__':
    res = add_num('199', '99')
    print(res)
        