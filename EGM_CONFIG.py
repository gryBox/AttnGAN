
LEARNING_MAP_CONFIG = {
    "VERSION_LENGTH": "Short", # "Alternative is 'Long'Determines wether to do semantic search
    "LMAP_RUN_MODE":  "Loop", # Alternative is "Parallel"  Determines if to make and lmap (loop) or run as node for parallel use
    "DELETE_ENDTERM": True, # Secomd order Deletes records for the user inputerm and topic
    "MAX_NUMBER_KEYWORDS": 3, # Can be fraction (as percent of text)
    "MAX_NUMBER_NODES": 1,
    "NODE_TABLE":{
        "dbName":"dynamodb",
        "tableName":"NodeTable"
        },
    "RESOURCE_TABLE": {
        "dbName":"dynamodb",
        "tableName":"ResourceDocuments"
        },
    "URL_TABLE": {},
    "GRAPH_DB":{},
    "NODE_IDENTIFIER_KEYS": ["inputTerm", "ancestorTerm", "userInputTerm", "Topic"],
    "NODE_STATUS_KEYS": ['SUCCESSFULLY_COMPLETE', "TO_BE_COMPLETE", "IN_PROGRESS", "FAILED"],
        
    "WRITE_TO_S3": True # NOT IMPLEMENTED - PASS DATA BETWEEN OBJECTS or FILES between functions
    }
# 'SUCCESSFULLY_COMPLETE', "TO_BE_COMPLETE", "IN_PROGRESS", "FAILED"
GRADE_LEVEL_COLUMNS = [
    'automated_readability_index',
    'coleman_liau_index',
    'flesch_kincaid_grade_level',
    'gunning_fog_index',
    'smog_index',
    "wiener_sachtextformel"
    ]
    
TEXT_DIFFICULTY_COLUMNS = [
    'flesch_reading_ease',
    'lix'
    ]


DEFINITION_SOURCES = ['Wordnik_Definitions']

NODE_STORE_COLS = [
    "NODE_IDENTIFIER",
    "TERM",
    "TOPIC",
    "ANCESTORE_TERM",
    "ANCESTORE_NODE",
    "KEYWORD_SCORE",
    "NODE_STATUS",
    "POS",
    "NUMBER_OF_TEXTS",
    "BASIC_COUNTS",
    "GRADE_LEVEL_STATS",
    "READABILITY_STATS",
    "END_TERM",
    "END_TOPIC"
    ]

RESEARCH_DOCUMENT_STORE_COLS = [
    "NODE_IDENTIFIER", 
    "TERM", 
    "RESOURCE",
    "RESOURCE_URL",
    "RESOURCE_SOURCE",
    "RESOURCE_DATATYPE",
    "RESOURCE_ATTRIBUTION", 
    "POS",
    'RESOURCE_TYPE',
    "UNIQUE_IDENTIFIER",
    "TIME_DOWNLOADED",
    "IMAGES",
    "TOPIC",
    "READABILITY_STATS",
    "BASIC_COUNTS"
    "END_TERM",
    "END_TOPIC"
    ]

WORDNIK_CONFIG = {
    "API_KEY" : "fab7b1dee9af0593b1005006e2e029b33df5f539e1e8f9c4f",
    "BASE_URL" : "http://api.wordnik.com/v4",
    "DEFINTION_DOCUMENT_STORE_COLMAP":{
        "word" : RESEARCH_DOCUMENT_STORE_COLS[1],
        "text" : RESEARCH_DOCUMENT_STORE_COLS[2],
        "sourceDictionary" : RESEARCH_DOCUMENT_STORE_COLS[4],
        "attributionText" : RESEARCH_DOCUMENT_STORE_COLS[6],
        "partOfSpeech" : RESEARCH_DOCUMENT_STORE_COLS[7]
        },
    "EXAMPLES_DOCUMENT_STORE_COLMAP":{}
}

WIKIPEDIA_CONFIG = {
    "wiki_url_base": "https://en.wikipedia.org/wiki/",
    "DOCUMENT_STORE_COLMAP" :{
        "references":"REFERENCES",
        "summary": RESEARCH_DOCUMENT_STORE_COLS[2],
        "images": RESEARCH_DOCUMENT_STORE_COLS[11],
        "url": RESEARCH_DOCUMENT_STORE_COLS[3],
        "content": RESEARCH_DOCUMENT_STORE_COLS[2]
        
    }
}

DUCKDUCKGOAPI_CONFIG = {
    "RESOURCE_TYPE": "secondary",
    "DOCUMENT_STORE_COLMAP" :{
        "AbstractSource":RESEARCH_DOCUMENT_STORE_COLS[4],
        'AbstractText': RESEARCH_DOCUMENT_STORE_COLS[2],
        "AbstractURL": RESEARCH_DOCUMENT_STORE_COLS[3],
    }
}

S3_NODE_DATASTORE_CONFIG = {
    "bucketName":"egm-bucket",
    "datastoreDir":"NODE-DATASTORE-TMP"
}

DYNAMODB_NODE_RESOURCE_CONFIG = {
    "RESOURCE_TABLE_NAME": "ResourceDocuments"
}

# Definition Providers with API use