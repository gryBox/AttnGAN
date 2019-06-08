"""
Calculates Stats on a corpus.  A describe corpus similar to pandas.dataframe.describe
TODO: Add max and other stats.  Should this be extended to include those docs?
"""
import textacy
# doc extentions are a concept introduced in textacy 0.7.0
textacy.spacier.doc_extensions.set_doc_extensions()

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CorpusStats:
    def __init__(self, corpus):
        """
        A minimal set of statistical descriptions needed to maximize_captions.
        TODO:  Add other stats e.g. max_sents, max_tokens
        """

        self.min_sents = self.minSents(corpus)
        self.min_tokens = self.minTokens(corpus)

        logger.debug(f"Finished calculating CorpusStats")

    def minSents(self, corpus):
        min_sents = min([doc._.n_sents for doc in corpus])
        return min_sents

    def minTokens(self, corpus):
        min_tokens = min([doc._.n_tokens for doc in corpus])
        return min_tokens