'''
Code to extract key phrase from input text usning python functions in spark
Using textacy and Gensim
By: sandipan ghosh

'''

from pyspark.sql import SparkSession
import string
import unidecode
import re
import time
import spacy
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import textacy.ke
from textacy import *
from pyspark.sql.functions import udf, struct, col
from pyspark.sql.types import * 
import pyspark.sql.functions as func
from pyspark.sql.functions import *

import os
from pyspark.sql.types import StringType, ArrayType

# ---------------------------------------------------------------------- #
spark = SparkSession.builder \
        .appName("get_keywords") \
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
       _model = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))
    en = _model
    return en

# input file path
file_path = "/root/docker_data/NLP/data/output/pre_process_twitter_data_lamma/*.json"

def remove_url(text):
    # Remove any web url starting with http or www
    text = 'NULL' if text is None else re.sub(r'(http|https|ftp|ssh|sftp|www)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    return text

remove_url_udf = udf(lambda row: remove_url(row), StringType())
spark.udf.register("remove_url_udf", remove_url, StringType())

def remove_all_punctuation(text):
  '''Remove other punctuation, adding fe more
  string.punctuation = !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
  '''
  PUNCT_TO_REMOVE = string.punctuation
  text = 'NULL' if text is None else str(text).translate(str.maketrans('', '', PUNCT_TO_REMOVE))
  return text

remove_all_punctuation_udf = udf(lambda row: remove_all_punctuation(row), StringType())
spark.udf.register("remove_all_punctuation_udf", remove_all_punctuation, StringType())

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = 'NULL' if text is None else unidecode.unidecode(str(text))
    return text

remove_accented_chars_udf = udf(lambda row: remove_accented_chars(row), StringType())
spark.udf.register("remove_accented_chars_udf", remove_accented_chars, StringType())

def to_lower(text):
  '''conver all to lower'''
  text = 'NULL' if text is None else str(text).lower()
  return text

to_lower_udf = udf(lambda row: to_lower(row), StringType())
spark.udf.register("to_lower_udf", to_lower, StringType())

def remove_tabs(input):
    text = 'NULL' if input is None else str(input)
    '''remove all the tab, new line char'''
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    return text

remove_tabs_udf = udf(lambda row: remove_tabs(row), StringType())
spark.udf.register("remove_tabs_udf", remove_tabs, StringType())

def remove_blanks(text):
  '''remove all the more than 1 spaces'''
  text = 'NULL' if text is None else re.sub(' +', ' ', str(text))
  return text

remove_blanks_udf = udf(lambda row: remove_blanks(row), StringType())
spark.udf.register("remove_blanks_udf", remove_blanks, StringType())

# extract keywords using textacy
def get_textacy_keywords(text):
    en = get_spacy_model()
    text = str(text)
    doc = textacy.make_spacy_doc(text, lang=en)
    Textrank = [kps for kps, weights in textacy.ke.textrank(doc, normalize="lemma", topn=5)] #use this
    SGRank = [kps for kps, weights in textacy.ke.sgrank(doc, topn=5, normalize="lemma")] #use this
    return [str(Textrank),str(SGRank)]

get_textacy_keywords_udf = udf(lambda row: get_textacy_keywords(row), StringType())
spark.udf.register("get_textacy_keywords_udf", get_textacy_keywords, StringType())

# extract keywords using gensim
'''for small text using lemmatize = True did not result into a good performace. print(keywords(text_en,words = 10,scores = True, lemmatize = True))
using already lemmatize text does improve the performace.
example: 
>>> print(keywords("blacks dont call it the chinese virus thats racist   china", words=5,lemmatize = True))
virus thats racist
blacks dont
>>> print(keywords("black do not call it the chinese virus that s racist china", words=5,lemmatize = False))
china
chinese virus
'''
def get_gensim_keywords(text):
    text = str(text)
    keys = keywords(text,words=5,lemmatize = True).split('\n')
    return keys

get_gensim_keywords_udf = udf(lambda row: get_gensim_keywords(row), StringType())
spark.udf.register("get_gensim_keywords_udf", get_gensim_keywords, StringType()) 
# ---------------------------------------------------------------------- #

# read the json file
temp = spark.read.json(file_path)
# register the table
temp.createOrReplaceTempView("temp")
# count the rows
temp.count()

# clean_data = data['tweet_text'].apply(remove_all_punctuation)
# start_time = time.time()
# temp_clean = spark.sql("""select *, remove_blanks_udf(
#                                     remove_tabs_udf(
#                                     to_lower_udf(
#                                     remove_accented_chars_udf(
#                                     remove_all_punctuation_udf(
#                                     remove_url_udf(
#                                                     tweet_text
#                                                     )))))) as clean_tweet_ie
#                             from temp""")
# temp_clean = temp_clean.withColumn("clean_tweet_text_lamma", get_base_word_udf("clean_tweet"))
# temp_clean.createOrReplaceTempView("temp_clean")
# execute the above statement
# temp_clean.count()
# print("--- %s seconds ---" % (time.time() - start_time))

# extract key words
start_time = time.time()
temp_clean_keywords = spark.sql("""select *, get_textacy_keywords_udf(clean_tweet_text_lamma) as textacy_keywords,
                                  get_gensim_keywords_udf(clean_tweet_text_lamma) as gensim_keywords
                                  from temp""")
# temp_clean_keywords.where("id = 1319485519462543361").show(1,False)
# write to a json
temp_clean_keywords.write.json("/root/docker_data/NLP/data/output/keywords_lg")
print("--- %s seconds ---" % (time.time() - start_time))


'''
issues :-
with lg - OSError: [E050] Can't find model 'en_core_web_lg.vectors'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

https://github.com/explosion/spaCy/issues/3552

'''