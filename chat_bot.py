import io
import nltk
import numpy as np
import random
import string #processes standard python strings
# converts a collection of raw documents to a matrix of features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# open and convert all text to lowercase for pre-processing
file = open('chat_bot.txt', 'r', encoding ='utf8', errors = 'ignore')
raw = file.read()
raw = raw.lower()

# first time use only
# nltk.download('punkt') # uncomment for first-time use only
# nltk.download('wordnet') # uncomment for first-time use only

# converts to list of sentences
sent_tokens = nltk.sent_tokenize(raw) 
# converts to list of words
word_tokens = nltk.word_tokenize(raw) 


# Semantically-oriented dictionary in NLTK
lemmer = nltk.stem.WordNetLemmatizer() 

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword matching

GREETING_INPUTS = ("hello", "hi", "greetings", "what's up", "hey")

GREETING_RESPONSES = ["Hi", "Hey", "*nods*", "Hi there", "Hello"]

# if user's input is greeting, returns greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# searches the user's phrase for general keywords and returns a response
def response(user_response):
    chat_bot_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf==0):
        chat_bot_response = chat_bot_response + "Sorry, I don't know that."
        return chat_bot_response
    else:
        chat_bot_response = chat_bot_response + sent_tokens[idx]
        return chat_bot_response

name = "BeatleBot"
subject = "The Beatles"

# starting and ending phrases
flag=True
print(name + ": Hi, I'm " + name + ". I will answer all your questions about " + subject + ". If you want to exit, type bye!")

while(flag==True):
    user_response = input("Me: ")
    user_response = user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print(name + ": You are welcome! Have a great day!")
        else:
            if(greeting(user_response)!=None):
                print(name + ": " + greeting(user_response))
            else:
                print(name + ": ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print(name + ": Bye! Have a great day!")

