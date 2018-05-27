import spacy
import numpy as np
from pyphen import Pyphen


class Readability:
    """Readability score."""

    def __init__(self, spacy_model='en_core_web_sm', pyphen_language='en_US'):
        self.pyphen_language= pyphen_language
        self.spacy_model = spacy_model
        self.text = ''
        self.text_nlp = False
        self.asl = 0
        self.asw = 0
        self.fre = 0

    def _count_syllables(self, word):
        """Counting the syllables in a word."""
        dic = Pyphen(lang=self.pyphen_language)
        word = dic.inserted(word)
        s_count = word.count("-") + 1

        return s_count

    def _get_asw(self):
        """Get the average number of syllables per word."""

        if not self.text_nlp:
            return False

        s_lengths = []
        for token in self.text_nlp:
            s_lengths.append(self._count_syllables(token.text))

        return np.array(s_lengths).mean()

    def _get_asl(self):
        """Get the average sentence length."""

        if not self.text_nlp:
            return False

        s_lengths = []
        for sent in self.text_nlp.sents:
            s_lengths.append(len(sent))

        return np.array(s_lengths).mean()

    def _get_flesh_reading_ease(self):
        """Get the Flesh-Reading-Ease score."""

        if not self.text_nlp:
            return False

        return 206.835 - (1.015 * self.asl) - (84.6 * self.asw)


    def score(self, text):
        """Score a given text."""
        # Basic NLP
        self.text = text.strip()
        self.nlp = spacy.load(self.spacy_model)
        self.text_nlp = self.nlp(self.text)

        # Basic Statistics
        self.asw = self._get_asw()
        self.asl = self._get_asl()

        # Reading Scores
        self.fre = self._get_flesh_reading_ease()

        scores = {'Flesch-Reading-Ease': self.fre}

        return scores