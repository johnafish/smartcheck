"""
Smartcheck: smart spellcheck in pure Python.

 FEATURES:
  - n-gram language model 

 TODO:
  - Contextual spellcheck from emails (enron corpus)
  - Neural language model
"""

from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from random import random

class Smartcheck:
    """A smart spell checker.

    Uses an 3-gram language model.
    """

    def __init__(self, corpus):
        """Initializes model."""
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self.corpus = corpus
        
        for sentence in self.corpus.sents():
            for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
                self.model[(w1, w2)][w3] += 1

        for wp in self.model:
            total_count = float(sum(self.model[wp].values()))
            for w3 in self.model[wp]:
                self.model[wp][w3] /= total_count

    def predict(self, sentence):
        words = sentence.split()[-2:]

        options = dict(self.model[(words[0], words[1])])
        threshold = random()
        accumulator = 0.

        for option, probability in options.items():
            accumulator += probability
            if accumulator >= threshold:
                return option
        return ""


if __name__ == "__main__":
    sc = Smartcheck(reuters)
    sentence = "John Fish is the"

    for i in range(10):
        sentence += " {}".format(sc.predict(sentence))

    print(sentence)
