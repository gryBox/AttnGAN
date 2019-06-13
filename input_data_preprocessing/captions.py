"""
A module for preprocessing text for text to image algorithms. The functions are primarily used for controling the number of captions per image for a text.

"""
import pandas as pd
import textacy
import numpy as np
import en_core_web_sm

import input_data_preprocessing as idp

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ReshapeImageLabels():
    """
    Reshape all the image labels to have the same number of captions while trying maintain an even  num of characters per caption.
    """
    def __init__(self, captionsCorpus):

        # Input - A Spacy corpus of texts corresponding to a set of images
        logging.info(f"Input Corpus: {captionsCorpus}")

        # 1  Calculate corpus stats
        self.corpusStats = idp.corpus_stats.CorpusStats(captionsCorpus)

        # 2  Extract the shortest doc from a corpus
        self.shortestDoc = idp.utils.find_shortest_doc(captionsCorpus, self.corpusStats.min_tokens)

        # 3  Maximize the shortest doc captions and return a list of captions greater than entered
        self.shortestDocCaptions = MaximizeDocCaptions(self.shortestDoc)

        # 4  Reshape all the image labels to have the same number of captions while trying maintain an even
        self.image_label_list = self.shape_text_captions(captionsCorpus, self.shortestDocCaptions.num_of_captions)


    def shape_text_captions(self, captionsCorpus, max_captions):
        """
        Reshape all the image labels to be of the same number of rows, with an attempt keep the number of characters the same for each image.
        TODO: Filter captions by similarity or information or length
        """
        corpus_captions_lst = list()
        to_shortDocs_lst = list()

        # d. Loop over corpus and re-size labels to the max captions i.e ideally split by sentences
        for docidx, doc in enumerate(captionsCorpus):
            logging.debug(f"New doc in Corpus Resizing Image Caption {docidx}")
            image_captions_lst = list()
            # Check if the doc the labaels need to be minimized or expanded
            if doc._.n_sents==max_captions:

                # Split sents into captions and noramlize text i.e lowercase everything.
                image_captions_lst = [normalize_caption_text(sent.text) for sent in doc.sents]

            elif doc._.n_sents<max_captions:

                # Maximize the captions per image
                maxedCaptions = MaximizeDocCaptions(doc)

                # Check that maxed captions list is not larger than the max captions allowed from the shortest doc
                if len(maxedCaptions.captions_lst)>max_captions:

                    # Minimize the new reshaped captions to the proper number of captions i.e. max captions
                    minnedCaptions = MinimizeDocCaptions(maxedCaptions.captions_lst, max_captions)
                    image_captions_lst = minnedCaptions.captions_lst
                else:
                    image_captions_lst = maxedCaptions.captions_lst

            elif doc._.n_sents>max_captions:

                # Split sentences into list  before munging
                captions_lst = [sent.text for sent in doc.sents]

                minnedCaptions = MinimizeDocCaptions(captions_lst, max_captions)
                image_captions_lst = minnedCaptions.captions_lst

            logging.debug(f"Final Number of captions per image: {len(image_captions_lst)}")
            corpus_captions_lst.append(image_captions_lst)

        return corpus_captions_lst


class MaximizeDocCaptions():
    """
    Attempt to maximize captions for text.  Takes a textacy doc and returns a list of captions.
    Notes:
        Normalization of captions happens in the function that maximizes sent(s)
    """
    def __init__(self, doc):

        self.captions_lst = self.maximize_captions(doc)
        self.num_of_captions = self.number_of_captions(self.captions_lst)
        logging.info(f"MaximizeDocCaptions - Number of captions {self.num_of_captions}")

    def maximize_captions(self, doc):


        captions_lst = list(doc._.to_terms_list(ngrams=(3, 4, 5),
            normalize = 'lower',
            entities=True,
            weighting="binary",
            filter_punct = True,
            drop_determiners = True,
            as_strings=True)
            )

        logging.debug(f"Finished - Maximizimizing number of captions per image")

        return captions_lst

    def number_of_captions(self, captions_lst):

        return len(captions_lst)


def normalize_caption_text(text):
    """
    Once the captions are chosen, they have to be all normalized for the model i.e so they all look the same
    """
    # input_string_bool = type(text)==str
    # Sprint(f"Is of type string: {input_string_bool}")
    caption = textacy.preprocess.preprocess_text(text,
                                                 fix_unicode=False,
                                                 lowercase=True,
                                                 no_urls=True,
                                                 no_emails=True,
                                                 no_phone_numbers=True,
                                                 no_numbers=False,
                                                 no_currency_symbols=False, no_punct=True,
                                                 no_contractions=False,
                                                 no_accents=False)
    return caption

# Take a label for an image and split into a defined number of captions
class MinimizeDocCaptions():
    """
    Reshapes captions to the desires number allowed by max captions.
    Notes:

    """
    def __init__(self, captions_lst, max_captions, normalize_text=True, captions_clm_name="captions"):

        # Max captions for the whole corpus
        self.max_captions = max_captions

        self.captions_clm_name = captions_clm_name

        captions_df = idp.utils.txt_to_df(captions_lst, captions_clm_name=captions_clm_name)

        if normalize_text:
            captions_df[captions_clm_name] = captions_df[captions_clm_name].apply(normalize_caption_text)

        # Calculate the ideal length a caption should be
        self.ideal_caption_length = int(np.ceil(captions_df['n_chars'].sum()/max_captions))
        logging.debug(f"Ideal caption length: {self.ideal_caption_length}")


        self.captions_lst = self.segment_captions(captions_df, self.max_captions, self.ideal_caption_length)


    def segment_captions(self, captions_df, max_captions, ideal_caption_length):
        """
        Loops through dataframe and concats the sents by cum summing the n_chars for each sent and aggregating until all the sents fall into a bin
        """

        captions_list = list()
        while len(captions_list)!=max_captions:

                # cumsum n_chars to find cutoff threshhold
                captions_df['cumsum_chars'] = captions_df['n_chars'].cumsum()

                # Make a new dataframe of chosen sentence structures to concatenate
                new_caption_df = captions_df.loc[captions_df['cumsum_chars']<=ideal_caption_length, self.captions_clm_name]

                # Concat all the string in the filterd df to one caption in a list
                caption_str = new_caption_df.str.cat(sep=' ')

                # drop the rows chosen to create a new caption
                captions_df = captions_df.drop(new_caption_df.index, axis=0).reset_index(drop=True).copy()

                # Handle case when the captions cannot be concatenated to max captions
                if (len(captions_list)==max_captions) and (captions_df.shape[0]>0):

                    # Concat all the string in the filterd df to one caption in a list
                    remainder_captions_str = captions_df[self.captions_clm_name].str.cat(sep=' ')
                    caption_str = f"{caption_str} {remainder_captions_str}"

                logging.debug(f"Number of characters per captions: {len(caption_str)}")
                captions_list.append(caption_str)

        return captions_list



