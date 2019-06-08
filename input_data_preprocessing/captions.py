"""
A module for preprocessing text for text to image algorithms. The functions are primarily used for controling the number of captions per image for a text.

"""
import pandas as pd
import textacy
import numpy as np
import en_core_web_sm


import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def reshape_corpus_captions():
    """
    Reshape all the image labels to be of the same number.
    TODO: Filter captions by similarity or information or length
    """


    return


class MaximizeDocCaptions():
    """
    Attempt to maximize captions for text.  Takes a textacy doc and returns a list of captions.
    Notes:
        Normalization of captions happens in the function that maximizes sent(s)
    """
    def __init__(self, doc):

        self.caption_lst = self.maximize_captions(doc)
        self.num_of_captions = self.number_of_captions(self.caption_lst)
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
    def __init__(self, captions_df, max_captions, normalize_text=True, captions_clm_name="captions"):

        # Max captions for the whole corpus
        self.max_captions = max_captions

        self.captions_clm_name = captions_clm_name

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

                print(f"Number of characters per captions: {len(caption_str)}")
                captions_list.append(caption_str)

        return captions_list



