import nltk
import random
import cPickle
import sys, os
import json
from glob import glob

from utils import num_wiggle, parse_weight_string, weighted_avg, join_dicts, retokenize, make_model_fname, make_stubs_fname
from train_model import build_model

GENERATOR_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(GENERATOR_ROOT, os.pardir))
LANG_MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
TEXT_DIR = os.path.join(PROJECT_ROOT, 'data', 'sotus')


from bs4 import BeautifulSoup

class Obama(object):
    _stats = {
        'preambles': [
            "Madam Speaker, Vice President Biden, Members of Congress, distinguished guests, and fellow Americans:", 
            "Mr. Speaker, Mr. Vice President, Members of Congress, distinguished guests, and fellow Americans:", 
            "Please, everybody, have a seat. Mr. Speaker, Mr. Vice President, Members of Congress, fellow Americans:", 
            "Madam Speaker, Mr. Vice President, Members of Congress, the First Lady of the United States--she's around here somewhere:", 
            "Mr. Speaker, Mr. Vice President, Members of Congress, distinguished guests, and fellow Americans:", 
            "Mr. Speaker, Mr. Vice President, Members of Congress, my fellow Americans"
        ],
        "avg_sent_length": 15,
        "avg_speech_length": 27,
        "avg_para_length": 5
    }

    def __init__(self, ngram_pickle):
        if isinstance(ngram_pickle, basestring):
            _ngram_file = open(ngram_pickle)
        else:
            _ngram_file = ngram_pickle
        self._ngram_model = cPickle.load(_ngram_file)
        self.ngram_order = self._ngram_model._n
        self.window_length = 3

    @property
    def avg_para_length(self):
        return self._stats['avg_para_length']

    @property
    def avg_speech_length(self):
        return self._stats['avg_speech_length']

    @property
    def preamble(self):
        text = random.choice(self._stats['preambles'])
        soup = BeautifulSoup(text)
        tokens = nltk.word_tokenize(soup.get_text())
        tokens.append('~SENT~')
        return tokens

    def next_word(self, context):
        result = self._ngram_model.generate(1, context=context)
        window = min([self.window_length, len(result)])
        return (result[-window:], result[-1])

    def next_sent(self, context=None):
        sent = []
        while len(sent) < 2:
            sent = self.make_sent(context)
        return sent

    def make_sent(self, context):
        sent = []
        _context = context
        if not context:
            _context = self.preamble
            sent.extend(_context)
            return sent
        nw = ""
        while nw != '~SENT~':
            _context, nw = self.next_word(_context)
            sent.append(nw)
        return sent


def make_obama(ngram_order=3, select=None):
    sys.stderr.write('assembling obama...\n')
    model_fname = make_model_fname(ngram_order)
    model_loc = os.path.join(LANG_MODEL_DIR, model_fname)
    # if not os.path.exists(model_loc):
    #     print model_loc
    #     raise Exception('no {order}-gram model found for Obama)'.format(
    #         order=ngram_order))
    build_model(ngram_order)
    obama = Obama(model_loc)
    sys.stderr.write('loaded.\n')
    return obama





class SpeechWriter(object):
    def __init__(self, ngram_order=3, randomize=False):
        self._obama = make_obama(ngram_order)
        self.speech_stats = {
            "avg_sent_length": 15,
            "avg_speech_length": 27,
            "avg_para_length": 5
        }


if __name__ == "__main__":
    sys.stderr.write("Starting demo...\n")
    sys.stderr.write("="*80+'\n')
    
    sw = SpeechWriter()
    speech = sw.generate_speech()

    sys.stderr.write("One Paragraph...\n\n")
    print speech.next()

