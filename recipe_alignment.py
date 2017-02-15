#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import nltk
import gensim
import cPickle as pickle


model = gensim.models.Word2Vec.load_word2vec_format('/home/meerahahn/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
from nltk.parse.stanford import StanfordDependencyParser
path_to_jar = \
    './stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'
path_to_models_jar = \
    './stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar,
        path_to_models_jar=path_to_models_jar)

def parseQuery(sentence):
    text = nltk.word_tokenize(sentence)
    pos_list = nltk.pos_tag(text)
    result = dependency_parser.raw_parse(sentence)
    dep = result.next()
    dep_list = list(dep.triples())
    return (pos_list, dep_list)


def parse_recipe(path):
    direct_objects = []
    with open(path) as r:
        content = r.readlines()
        for i in content:
            if len(i) == 2:
                continue
            result = dependency_parser.raw_parse(i)
            dep = result.next()
            dep_list = list(dep.triples())
            # check = False
            for dependency in dep_list:
                if dependency[1] == 'dobj':
                    tup = [dependency[0][0], dependency[2][0]] 
                    direct_objects.append((tup, i))
                    # check = True
            # if not check:
            #     text = nltk.word_tokenize(i)
            #     pos_list = nltk.pos_tag(text)
    return direct_objects

def get_object_vectors(direct_objects):
    object_vectors  = []
    for (o, d_o) in enumerate(direct_objects):
        object = (d_o[0][1])
        object_vectors.append(model[object])
    return object_vectors

def get_annotations(path, direct_objects):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for each in content:
        temp = each.split('>')
        for i in range(3):
            temp[i] = temp[i].replace('<', '')
        temp[1] = temp[1].split(',')
        if temp[0] == 'read':
            continue
        object = temp[1][0]
        check = False
        if object not in model.vocab:
            if '_' in object:
                object = object.split('_')[0]
        best_sim = -999.0
        index = 0
        line = ''
        for (ind, do) in enumerate(direct_objects):
            dist = model.similarity(do[0][1], object)
            if dist > best_sim:
                index = ind
                best_sim = dist
                line = do[1]
        print index
        print direct_objects[index]
        print object
        print temp
        print line
        print 

if __name__ == '__main__':
    recipe_name = 'north_american_breakfast'
    direct_objects = \
         parse_recipe('./recipies/' + recipe_name)
    print direct_objects
    #object_vectors = get_object_vectors(direct_objects)
    get_annotations('./labels_cleaned/Subject_2/Carlos_American.txt', direct_objects)


			