'''
Code to extract NER using spacy from input text usning python functions in spark
By: sandipan ghosh
Date: 20/11/2020
'''

from pyspark.sql import SparkSession
import string
import unidecode
import re
import time
import spacy
from pyspark.sql.functions import udf, struct, col
from pyspark.sql.types import * 
import pyspark.sql.functions as func
from pyspark.sql.functions import *

import os
from pyspark.sql.types import StringType, ArrayType

# ---------------------------------------------------------------------- #
spark = SparkSession.builder \
        .appName("get_ner") \
        .getOrCreate()
        #.master("spark://05d64dcca62a:7077") \
# ---------------------------------------------------------------------- #
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# spark.conf.set("spark.python.worker.reuse", "true")
# spark.conf.set("spark.python.worker.memory", "2g")
# ---------------------------------------------------------------------- #
# Load the en_core_web_sm
# en = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))
# en = textacy.load_spacy_lang("en_core_web_lg", disable=("parser",))
en = None
def get_spacy_model():
    global en
    global _model
    if not en:
       _model = spacy.load('en_core_web_lg')
    en = _model
    return en

# input file path
file_path = "/root/docker_data/NLP/data/output/pre_process_twitter_data_lemma/*.json"

# Get all NER using SPACY
def get_ner_spacy(text):
    text = 'NULL' if text is None else str(text)
    nlp = get_spacy_model()
    ner_tags = {}
    if text:        
        doc = nlp(text)
        for ee in doc.ents:
            ner_tags[ee.text] = ee.label_
    return ner_tags

get_ner_spacy_udf = udf(lambda row: get_ner_spacy(row), StringType())
spark.udf.register("get_ner_spacy_udf", get_ner_spacy, StringType())

# get all NOUNs and proper nouns POS
def get_noun_spacy(text):
    text = 'NULL' if text is None else str(text)
    nlp = get_spacy_model()
    noun_tags = {}
    if text:        
        doc = nlp(text)
        for token in doc:
            if(token.pos_ == 'NOUN' or token.pos_ == 'PROPN'):
                noun_tags[token.text] = token.pos_
    return noun_tags

get_noun_spacy_udf = udf(lambda row: get_noun_spacy(row), StringType())
spark.udf.register("get_noun_spacy_udf", get_noun_spacy, StringType())

# read the json file
temp = spark.read.json(file_path)
# register the table
temp.createOrReplaceTempView("temp")
# count the rows
# temp.count()
# get the NER and POS
temp_clean_keywords = spark.sql("""select *, get_ner_spacy_udf(clean_tweet_text_lemma) as spacy_ner,
                                  get_noun_spacy_udf(clean_tweet_text_lemma) as spacy_noun
                                  from temp""")