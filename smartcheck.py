"""
Smartcheck: smart spellcheck in pure Python.

 FEATURES:
  - Norvig's autocorrect
  - 3-gram language model 

 TODO:
  - Combine Norvig + 3-gram approaches
  - Build better error model with errors from text
  - Save + pickle the trained 3-gram, language, error models
"""

from nltk import trigrams, word_tokenize
from collections import Counter, defaultdict
import re

class Smartcheck:
    """A smart spell checker.

    Uses an 3-gram language model.
    """

    def __init__(self, corpus):
        """Initializes language model with trigram probabilities."""
        self.corpus = corpus
        self.trigrams = defaultdict(lambda: defaultdict(lambda: 0))
        self.model = {} 
        self.pop_model()
        self.pop_trigrams()

    def sentences(self, text):
        """All sentences in a given text."""
        return re.findall(r'([A-Z][^\.!?]*[\.!?])', text)

    def words(self, text):
        """All words in a given text."""
        return re.findall(r'\w+', text)

    def pop_model(self):
        """Populate model with probability of word."""
        word_counts = Counter(self.words(self.corpus))
        N = sum(word_counts.values())
        for word in word_counts:
            self.model[word] = word_counts[word] / N
        
    def pop_trigrams(self):
        """Populate self.trigrams with probabilities of next words"""
        for sentence in self.sentences(self.corpus):
            for w1, w2, w3 in trigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
                self.trigrams[(w1, w2)][w3] += 1

        # Convert trigrams to probabilities
        for wp in self.trigrams:
            total_count = float(sum(self.trigrams[wp].values()))
            for w3 in self.trigrams[wp]:
                self.trigrams[wp][w3] /= total_count

    def predict(self, sentence):
        """Predict the next words given the sentence."""
        prev_two_words = sentence.split()[-2:]
        options = dict(self.trigrams[tuple(prev_two_words)])
        return options

    def word_probability(self, word):
        """Probability of a given word."""
        if word in self.model:
            return self.model[word]
        return 0

    def correction(self, word):
        """Return the most probable correction."""
        return max(self.candidates(word), key=self.word_probability)

    def candidates(self, word):
        """Candidate list of possible correct words."""
        return (self.known([word]) or \
                self.known(self.edits1(word)) or \
                self.known(self.edits2(word)) or \
                set([word]))

    def known(self, words):
        return set(w for w in words if w in self.model)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

if __name__ == "__main__":
    sentence = "This is a test sentence. This is another. This is a. Okay? Okay! Fine then."
    sc = Smartcheck(sentence)
    print(sc.correction("sentnce"))
