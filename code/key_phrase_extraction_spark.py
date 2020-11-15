from pyspark.sql import SparkSession
import string
import unidecode
import re
import time
import spacy
import textacy.ke
from textacy import *
from pyspark.sql.functions import udf, struct, col
from pyspark.sql.types import * 
import pyspark.sql.functions as func
from pyspark.sql.functions import *

en = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))

file_path = "/root/docker_data/NLP/data/*.json"

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

def get_ie_keywords(text):
    text = str(text)
    doc = textacy.make_spacy_doc(text, lang=en)
    Textrank = [kps for kps, weights in textacy.ke.textrank(doc, normalize="lemma", topn=5)] #use this
    SGRank = [kps for kps, weights in textacy.ke.sgrank(doc, topn=5, normalize="lemma")] #use this
    return [str(Textrank),str(SGRank)]

get_ie_keywords_udf = udf(lambda row: get_ie_keywords(row), StringType())
spark.udf.register("get_ie_keywords_udf", get_ie_keywords, StringType())

# read the json file
temp = spark.read.json(file_path)
# register the table
temp.createOrReplaceTempView("temp")
# count the rows
temp.count()

# clean_data = data['tweet_text'].apply(remove_all_punctuation)
start_time = time.time()
temp_clean = spark.sql("""select *, remove_blanks_udf(
                                    remove_tabs_udf(
                                    to_lower_udf(
                                    remove_accented_chars_udf(
                                    remove_all_punctuation_udf(
                                    remove_url_udf(
                                                    tweet_text
                                                    )))))) as clean_tweet_ie
                            from temp""")
temp_clean.createOrReplaceTempView("temp_clean")
# execute the above statement
temp_clean.count()
print("--- %s seconds ---" % (time.time() - start_time))

# extract real key words
start_time = time.time()
temp_clean_keywords = spark.sql("""select *, get_ie_keywords_udf(clean_tweet_ie) as ie_keywords from temp_clean""")
# write to a json
temp_clean_keywords.write.json("/root/docker_data/NLP/keywords/")
print("--- %s seconds ---" % (time.time() - start_time))
