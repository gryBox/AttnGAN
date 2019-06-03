import pandas as pd
import textacy
import en_core_web_sm 
import boto3

def df_to_corpus(df):
    # Load into textacy to delimit sentences
    img_labels = df.to_dict(orient="records")
    text_stream, metadata_stream = textacy.io.split_records(img_labels, 'RESOURCE')

    # Load english model
    en = en_core_web_sm.load()
    corpus = textacy.Corpus(lang=en, texts=text_stream, metadatas=metadata_stream)
    
    return corpus