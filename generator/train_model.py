# -*- coding: UTF-8 -*-

import sys
import os
import json
from glob import glob
import pickle
import pdb

from collections import defaultdict

import unicodedata

import nltk
from nltk.model.ngram import NgramModel
from nltk.probability import LidstoneProbDist

GENERATOR_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(GENERATOR_ROOT, os.pardir))
LANG_MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
TEXT_DIR = os.path.join(PROJECT_ROOT, 'data', 'sotus')

from utils import weighted_avg, make_model_fname, make_stubs_fname
from bs4 import BeautifulSoup

replace_chars = {
    '&deg;': ' degree',
    '&eacute;': 'e',
    '&frasl;': '/',
    '&lsquo;': "'",
    '&lt;': " < ",
    '&mdash;': " - ",
    '&Ocirc;': "O",
    '&Otilde;': "O",
    '&pound;': "pounds ",
    '&#8226;': ""
    # '\xc2\x97': ' - ',
    # '\xc2\x91': "'",
    # '\xc2\x92': "'",
    # '\xc2\x93': '"',
    # '\xc2\x94': '"',
    # '\xe2\x80\x98': "'",
    # '\xe2\x80\x99': "'",
    # '\xe2\x80\x9a': "'",
    # '\xe2\x80\x9b': "'",
    # '\xe2\x80\x9c': '"',
    # '\xe2\x80\x9d': '"',
    # '\xe2\x80\x9f': '"',
    # '\xe2\x80\x9e': '"',
    # '\x60\x60': '"',
}

def est(fdist, bins):
    return LidstoneProbDist(fdist, 0.2)


def tokenize_and_demarcate(raw_paragraph):
    sent_list = []
    for sent in nltk.sent_tokenize(raw_paragraph):
        add_sent = nltk.word_tokenize(sent)
        add_sent.append('~SENT~')
        sent_list.append(add_sent)
    #sent_list[-1][-1] = '~PARA~'
    return sent_list


def parse_transcript(transcript_filename):
    speech = []
    stubs = []
    with open(transcript_filename) as transcript:
        for line in transcript:
            soup = BeautifulSoup(line.strip())
            raw = soup.get_text()
            print raw
            for orig_char, new_char in replace_chars.iteritems():
                raw = raw.replace(orig_char, new_char)
            if raw:
                sent = tokenize_and_demarcate(raw)
                for s in sent:
                    stub = s[0:3]
                    if ('~SENT~' not in stub) and ('bless' not in stub) and (all([x.isalpha() for x in stub])):
                        stubs.append(stub)
                speech.extend(sent)
    return (speech, stubs)


def first_paragraphs(name, prez_id):
    print name, '-',prez_id
    for transcript_filename in build_corpus(name):
        with open(transcript_filename) as transcript:
            raw = ''
            while not raw:
                line = transcript.readline()
                raw = nltk.clean_html(line.strip())
            print raw+'\n'
    raw_input('\nTo continue press enter')


def build_corpus():
    return glob(os.path.join(TEXT_DIR, '*'))


def train_model(corpus, order):
    speeches = []
    for transcript_filename in corpus:
        speech, stubs = parse_transcript(transcript_filename)
        speeches.extend(speech)
    ngram_model = NgramModel(order, speeches, estimator=est)
    return (ngram_model, stubs)


def pickle_model(ngram_model):
    order = ngram_model._n
    model_fname = make_model_fname(order)
    # import ipdb; ipdb.set_trace()
    with open(os.path.join(LANG_MODEL_DIR, model_fname), 'wb') as modelfile:
        pickle.dump(ngram_model, modelfile)


def pickle_stubs(stubs, prez_number):
    prez_number = unicode(prez_number).zfill(2)
    stubs_fname = make_stubs_fname(prez_number)
    with open(os.path.join(STUBS_DIR, stubs_fname), 'wb') as stubfile:
        pickle.dump(stubs, stubfile)


def avg_int(counts):
    return int(float(sum(counts)) / float(len(counts)))


def build_model(order):
    corpus = build_corpus()
    model, stubs = train_model(corpus, order)
    pickle_model(model)
    return model
