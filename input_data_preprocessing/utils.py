import pandas as pd
import boto3

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import textacy
import en_core_web_sm
textacy.spacier.doc_extensions.set_doc_extensions()

def df_to_corpus(df):
    # Load into textacy to delimit sentences
    img_labels = df.to_dict(orient="records")
    text_stream, metadata_stream = textacy.io.split_records(img_labels, 'RESOURCE')

    # Load english model
    en = en_core_web_sm.load()
    corpus = textacy.Corpus(lang=en, texts=text_stream, metadatas=metadata_stream)

    return corpus


def find_shortest_doc(corpus, n_tokens):
    # Find the doc with the minimum num of tokens.
    shortestDocs = corpus.get(lambda x: len(x)==n_tokens) # limit=1
    shortestDoc = list(shortestDocs)[0]

    logging.info(f"Shortest Doc - Number of tokens {shortestDoc._.n_tokens}")

    return shortestDoc



def txt_to_df(txt_lst, captions_clm_name="captions"):
    """
    Transform a list of texts to a df with some stats to reshape the number of captions
    """
    # Load the language model for textacy
    en = en_core_web_sm.load()

    captions_lst = list()
    for txt in txt_lst:

        doc = textacy.make_spacy_doc(txt, lang=en)

        ts = textacy.text_stats.TextStats(doc)
        df = pd.DataFrame({"n_chars" : [ts.n_chars],
                          captions_clm_name : doc.text})

        captions_lst.append(df)

    # Concat all df's into captions df for easy sorting and manipulation
    captions_df = pd.concat(captions_lst, ignore_index=True)
    captions_df['sent_order'] = captions_df.index

    return captions_df

