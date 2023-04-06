import nltk
import numpy as np
import random
import string #processes standard python strings

# open and convert all text to lowercase for pre-processing
file = open('chat_bot.txt', 'r', encoding ='utf8', errors = 'ignore')
raw = file.read()
raw = raw.lower()

nltk.download('punkt') # uncomment for first-time use only
nltk.download('wordnet') # uncomment for first-time use only

sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences
word_tokens = nltk.word_tokenize(raw) # converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()