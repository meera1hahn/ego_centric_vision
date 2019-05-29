#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys

def get_annotations(label_name):
    path = './labels_cleaned/' + label_name
    new_path = './new_annotations/' + label_name
    total = []
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for each in content:
        temp = each.split('>')
        for i in range(3):
            temp[i] = temp[i].replace('<', '')
        temp[1] = temp[1].split(',')
        temp[2] = temp[2].strip().replace('(', '').replace(')','').split('-')
        temp[2][0] = int(temp[2][0])
        temp[2][1] = int(temp[2][1])      
        total.append(temp)
    total = sorted(total, key=lambda annotation: annotation[2])
    print new_path
    new_f = open(new_path, 'w')
    for x in total:
        new_f.write(str(x)+'\n')
    new_f.close()

if __name__ == '__main__':
    label_name = 'Yin_Turkey.txt'
    get_annotations(label_name)
