from pyTextMiner.splitter import *
from pyTextMiner.tokenizer import *
from pyTextMiner.stemmer import *
from pyTextMiner.lemmatizer import *
from pyTextMiner.tagger import *
from pyTextMiner.helper import *
from pyTextMiner.counter import *
from pyTextMiner.chunker import *
from pyTextMiner.cooccurrence import *
from pyTextMiner.graphml import *
from pyTextMiner.ngram import *

from os import listdir
from nltk.corpus import stopwords
from pickle import dump
from string import punctuation
import os, sys
from stat import *
class Pipeline:
    def __init__(self, *pipelines):
        self.pipeline = pipelines
        self.collapse = self.checkType(pipelines)
        pass

    def checkType(self, pipeline):
        if not pipeline: return
        if pipeline[0].IN_TYPE != [str]: raise Exception("%s requires '%s' as input, but receives 'str'" % (type(pipeline[0]).__name__, pipeline[0].IN_TYPE))
        collapse = [[]]
        curType = pipeline[0].OUT_TYPE
        for p, q in zip(pipeline[:-1], pipeline[1:]):
            qt = q.IN_TYPE
            pt = p.IN_TYPE
            if qt == curType[-len(qt):]:
                collapse.append(curType[:-len(qt)])
                curType = curType[:-len(qt)] + q.OUT_TYPE
                continue
            raise Exception("%s requires '%s' as input, but receives '%s' from %s" % (type(q).__name__, qt, pt, type(p).__name__))
        return collapse

    def processCorpus(self, corpus):
        ''' process corpus through pipeline '''

        def apply(p, a, inst):
            if not a:
                return p(inst)
            if a[0] == list:
                return [apply(p, a[1:], i) for i in inst]
            if a[0] == dict:
                return {k:apply(p, a[1:], v) for k, v in inst}

        results = []
        for d in corpus:
            inst = d
            for p, c in zip(self.pipeline, self.collapse):
                inst = apply(p, c, inst)
            results.append(inst)
        return results

class Corpus:
    def __init__(self, textList):
        self.pair_map = {}
        self.docs = textList

    def __iter__(self):
        return self.docs.__iter__()

    def __len__(self):
        return self.docs.__len__()

class CorpusFromFile(Corpus):
    def __init__(self, file):
        self.docs = open(file, encoding='utf-8').readlines()

class CorpusFromFieldDelimitedFile(Corpus):
    def __init__(self, file, index):
        array = []
        with open(file, encoding='utf-8') as ins:
            for line in ins:
                array.append(line.split('\t')[index])

        self.docs = array

class CorpusFromFieldDelimitedFileWithYear(Corpus):
    def __init__(self, file, doc_index=1, year_index=0):
        array = []
        id = 0
        pair_map = {}
        with open(file, encoding='utf-8') as ins:
            for line in ins:
                fields = line.split('\t')
                array.append(fields[doc_index])
                pair_map[id] = fields[year_index]

                id += 1
        self.docs = array
        self.pair_map = pair_map

class CorpusFromDirectory(Corpus):

    def __init__(self, directory, is_train):
        array = []

        # walk through all files in the folder
        for filename in listdir(directory):
            # skip any reviews in the test set
            if is_train and filename.startswith('cv9'):
                continue
            if not is_train and not filename.startswith('cv9'):
                continue

            # create the full path of the file to open
            path = directory + '/' + filename
            # load the doc

            with open(path) as myfile:
                data = "".join(line.rstrip() for line in myfile)
            # add to list
            array.append(data)

        self.docs = array

