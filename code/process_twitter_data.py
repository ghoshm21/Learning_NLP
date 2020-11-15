'''
Documentation, License etc.
list of contractions:- https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
@package process_twitter_data
'''
from pyspark.sql import SparkSession
import json
from bs4 import BeautifulSoup
import unidecode
# from pycontractions import Contractions
import contractions
import pandas as pd
import re
import string
import requests
import spacy
from textblob import TextBlob
from spellchecker import SpellChecker
from gensim.models import KeyedVectors
#model = KeyedVectors.load('/root/docker_data/lib/gensim/GoogleNews-vectors-negative300', mmap='r')
#cont = Contractions(kv_model=model)
#cont.load_models()

#cont = Contractions('/root/docker_data/lib/GoogleNews-vectors-negative300.bin')
#cont.load_models()

# load spacy model
# nlp = spacy.load(r'/root/docker_data/lib/en_core_web_lg-2.3.1/en_core_web_lg/en_core_web_lg-2.3.1')
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

spark = SparkSession.builder \
        .getOrCreate()
        #.master("spark://05d64dcca62a:7077") \
        #.appName("process_twitter_data") \
       

from pyspark.sql.functions import udf, struct, col
from pyspark.sql.types import * 
import pyspark.sql.functions as func

from pyspark.sql.functions import *

# all data processing functions and UDF
def remove_html(input):
    """remove html tags from text"""
    '''Beautiful Soup ranks lxml’s parser as being the best, then html5lib’s, 
        then Python’s built-in parser.'''
    text = 'NULL' if input is None else str(input)
    # soup = BeautifulSoup(text, "html.parser")
    soup = BeautifulSoup(text, "lxml")
    stripped_text = soup.get_text(separator=" ", strip=True)
    return stripped_text

remove_html_udf = udf(lambda row: remove_html(row), StringType())
spark.udf.register("remove_html_udf", remove_html, StringType())

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = 'NULL' if text is None else unidecode.unidecode(text)
    return text

remove_accented_chars_udf = udf(lambda row: remove_accented_chars(row), StringType())
spark.udf.register("remove_accented_chars_udf", remove_accented_chars, StringType())

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

def remove_digits(text):
    # Remove digits, decimal numbers, dates and time format
    text = 'NULL' if text is None else re.sub(r'\d[\.\/\-\:]\d|\d', '', text)
    return text

remove_digits_udf = udf(lambda row: remove_digits(row), StringType())
spark.udf.register("remove_digits_udf", remove_digits, StringType())

def remove_all_punctuation(text):
    '''Remove other punctuation, adding fe more
    string.punctuation = !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
    '''
    PUNCT_TO_REMOVE = string.punctuation
    text = 'NULL' if text is None else text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    return text

remove_all_punctuation_udf = udf(lambda row: remove_all_punctuation(row), StringType())
spark.udf.register("remove_all_punctuation_udf", remove_all_punctuation, StringType())

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = 'NULL' if text is None else re.sub(pattern, '', text)
    return text

remove_special_characters_udf = udf(lambda row: remove_special_characters(row), StringType())
spark.udf.register("remove_special_characters_udf", remove_special_characters, StringType())

def ascii_to_string(text):
    # Encodes string to ASCII and decodes to string. This helps in removing any special characters in the database
    text = 'NULL' if text is None else  text.encode('ascii', 'replace').decode(encoding="utf-8")
    '''
    This replaces all special characters with a ?. Replacing this
    '''
    return text.replace('?', '')

ascii_to_string_udf = udf(lambda row: ascii_to_string(row), StringType())
spark.udf.register("ascii_to_string_udf", ascii_to_string, StringType())

def to_lower(text):
    '''conver all to lower'''
    text = 'NULL' if text is None else str(text).lower()
    return text

to_lower_udf = udf(lambda row: to_lower(row), StringType())
spark.udf.register("to_lower_udf", to_lower, StringType())

# def remove_url(text):
#     # Remove any web url starting with http or www
#     text = 'NULL' if text is None else re.sub(r'(www|http)\S+', '', text)
#     return text
def remove_url(text):
    # Remove any web url starting with http or www
    text = 'NULL' if text is None else re.sub(r'(http|https|ftp|ssh|sftp|www)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , text)
    return text

remove_url_udf = udf(lambda row: remove_url(row), StringType())
spark.udf.register("remove_url_udf", remove_url, StringType())

# def remove_email_address(text):
#     # Remove any email address
#     text = 'NULL' if text is None else re.sub(r'\S+@\S+', '', text)
#     return text
def remove_email_address(text):
    # Remove any email address
    text = 'NULL' if text is None else re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', '', text)
    return text

remove_email_address_udf = udf(lambda row: remove_email_address(row), StringType())
spark.udf.register("remove_email_address_udf", remove_email_address, StringType())

def get_email_address(text):
    # Remove any email address
    text = 'NULL' if text is None else str(text)
    email = [email for email in re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', text)]
    return email

get_email_address_udf = udf(lambda row: get_email_address(row), StringType())
spark.udf.register("get_email_address_udf", get_email_address, StringType())

def expand_contractions(text):
    text = 'NULL' if text is None else contractions.fix(str(text))
    return text

expand_contractions_udf = udf(lambda row: expand_contractions(row), StringType())
spark.udf.register("expand_contractions_udf", expand_contractions, StringType())

# count hashtag
def get_hashtag(data):
    data = 'NULL' if data is None else str(data)
    hashtag = [tag for tag in data.split() if tag.startswith('#')]
    return hashtag

get_hashtag_udf = udf(lambda row: get_hashtag(row), StringType())
spark.udf.register("get_hashtag_udf", get_hashtag, StringType())

# count mentions
def get_mention(data):
    data = 'NULL' if data is None else str(data)
    mention = [tag for tag in data.split() if tag.startswith('@')]
    return mention

get_mention_udf = udf(lambda row: get_mention(row), StringType())
spark.udf.register("get_mention_udf", get_mention, StringType())

# convert to base format except pronoun and be
def get_base_word(text):
    x = 'NULL' if text is None else str(text)
    x_list = []
    doc = nlp(x)
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
        x_list.append(lemma)
    return ' '.join(x_list)

get_base_word_udf = udf(lambda row: get_base_word(row), StringType())
spark.udf.register("get_base_word_udf", get_base_word, StringType())

# try Microsoft API
#  check spell using TextBlob
# TextBlob is fast and good enough
def correct_spell_textblob(text):
    text = 'NULL' if text is None else str(text)
    return str(TextBlob(text).correct())

correct_spell_textblob_udf = udf(lambda row: correct_spell_textblob(row), StringType())
spark.udf.register("correct_spell_textblob_udf", correct_spell_textblob, StringType())    

# check spell using PySpellCheck
# It's slow and chage the Propernoun
# Exmp: x = 'GFG is a good company and always value their employed.'
# GFG will converted to GIG
def correct_spell_spellcheck(text):
    text = 'NULL' if text is None else str(text)
    final = [spell.correction(word) for word in x.split()]
    return ' '.join(final)

correct_spell_spellcheck_udf = udf(lambda row: correct_spell_spellcheck(row), StringType())
spark.udf.register("correct_spell_spellcheck_udf", correct_spell_spellcheck, StringType())

# correct spell with enchant C lib 
# https://pyenchant.github.io/pyenchant/install.html
# https://www.tutorialspoint.com/get-similar-words-suggestion-using-enchant-in-python
# https://stackoverflow.com/questions/31026394/how-to-correct-text-and-return-the-corrected-text-automatically-with-pyenchant#new-answer
# need libenchant-dev C lib
# apt-get install libenchant-dev
# problem -  do not consider the context and try to match word by word.
# try 'outrageous top experts used by cdc move total us coronavirus deaths from  million to  to  in only  days httpstcofoxcgaacj'
# CDC will be CC and TextBlob will remove the CDC. This will chage the meaning of the line
# Try not to use it
import enchant, difflib
d = enchant.Dict("en_US")
from enchant.checker import SpellChecker
chkr = SpellChecker("en_US")
# my_sent="This is sme sample txt with erors httpstcofoxcgaacj."

def enchant_spellcheck(my_sent):
    my_sent = 'NULL' if my_sent is None else str(my_sent)
    chkr.set_text(my_sent)
    for err in chkr:
        dict,max = {},0
        error_word = err.word
        a = set(d.suggest(error_word))
        if len(a) > 0:                 
            for b in a:
                tmp = difflib.SequenceMatcher(None, error_word, b).ratio();
                dict[tmp] = b
                if tmp > max:
                    max = tmp
            # print(dict[max])
            err.replace(dict[max])
    c = chkr.get_text()#returns corrected text
    return c

enchant_spellcheck_udf = udf(lambda row: enchant_spellcheck(row), StringType())
spark.udf.register("enchant_spellcheck_udf", enchant_spellcheck, StringType())

# load the real data
data = spark.read.json("/root/docker_data/*.json")
data = data.withColumn("is_it_a_retweet",when(substring('text',1,2) == 'RT', 'Y').otherwise('N'))
# create the view for spark sql
data.createOrReplaceTempView("data")
# get only the required columns, to reduce the data size
filter_data_pre_process = spark.sql("""select distinct id, lang, created_at, source,
user.id_str as user_id_str, user.name as user_name,
user.location as user_location, user.description as user_description,
case when is_it_a_retweet = 'Y' and retweeted_status.truncated = 'true'
            then retweeted_status.extended_tweet.full_text
    when is_it_a_retweet = 'Y' and retweeted_status.truncated <> 'true'
            then retweeted_status.text
    when is_it_a_retweet <> 'Y' and truncated = 'true'
            then extended_tweet.full_text
    when is_it_a_retweet <> 'Y' and truncated <> 'true'
            then text
end as tweet_text
from data
where lang = 'en' """)

# create the temp view
filter_data_pre_process.createOrReplaceTempView("filter_data_pre_process")
# .where("id in (1319480915584864258, 1319480915735793666, 1319480914766974976)").show(5,False)

#filter_data.where("text is null").count()
#filter_data.where("quoted_status_text is null").count()
#filter_data.where("extended_tweet.full_text is null").count()

# filter_data_pre_process = filter_data.withColumn("tweet",coalesce(filter_data.full_text,filter_data.quoted_status_text, filter_data.text)) 
#filter_data_pre_process.repartition(200)
pre_process_twitter_data = spark.sql("""select id, lang, created_at, source,
user_id_str, user_name, user_location, user_description, tweet_text,
    get_hashtag_udf(tweet_text) as hashtag,
    get_mention_udf(tweet_text) as mention,
    get_email_address_udf(tweet_text) as emails,
    remove_all_punctuation_udf(
    to_lower_udf(
    remove_email_address_udf(
    remove_url_udf(
    remove_special_characters_udf(
    ascii_to_string_udf(
    remove_digits_udf(
    remove_blanks_udf(
    remove_tabs_udf(
    remove_accented_chars_udf(
    remove_html_udf(tweet_text)
    )))))))))) as clean_tweet_text
    from filter_data_pre_process""").where("clean_tweet_text is not null")
pre_process_twitter_data.createOrReplaceTempView("pre_process_twitter_data")
# write to the filesystem
pre_process_twitter_data.coalesce(16).write.format('json').save('/root/temp/clean_data')

# read the clean tweet
clean_tweet = spark.read.json("/root/temp/clean_data/*.json")
clean_tweet.createOrReplaceTempView("clean_tweet")
twitter_data_baseword = spark.sql("""select
id,
created_at,
emails,
hashtag,
lang,
mention,
source,
user_description,
user_id_str,
user_location,
user_name,
tweet_text,
clean_tweet_text,
get_base_word_udf(clean_tweet_text) as clean_tweet_text_lamma
from clean_tweet
""")

twitter_data_baseword.write.format('json').save('/root/temp/twitter_data_baseword')

process_twitter_data.count()

now = spark.read.json("/root/temp/clean_data/part-00000-f5bef4b9-a655-41a6-843d-d8f9772d561f-c000.json")
now.createOrReplaceTempView("now")

new_process = spark.sql("""select id, lang, created_at, source,
user_id_str, user_name, user_location, user_description, tweet_text,
    hashtag,
    mention,
    emails,
    get_base_word_udf(
    correct_spell_textblob_udf(tweet_text)) as clean_tweet_text
    from now""")
new_process.write.format('json').save('/root/temp/clean_data_word')




spark.sql("""select id, clean_tweet_text, 
            enchant_spellcheck_udf(clean_tweet_text) as enchant_spellcheck,
            correct_spell_textblob_udf(clean_tweet_text) as textBlob_spellcheck
    from now""").show(2,False)


dict,max = {},0
a = set(d.suggest(my_word))
for b in a:
   tmp = difflib.SequenceMatcher(None, my_word, b).ratio();
   dict[tmp] = b
   if tmp > max:
      max = tmp
print (dict[max])