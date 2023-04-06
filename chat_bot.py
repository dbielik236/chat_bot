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

nltk.download('popular', quiet=True) # for downloading packages

# open and convert all text to lowercase for pre-processing
file = open('chat_bot.text', 'r', encoding ='utf8', errors = 'ignore')
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

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")

GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

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
        chat_bot_response = chat_bot_response + "I am sorry! I don't understand you."
        return chat_bot_response
    else:
        chat_bot_response = chat_bot_response + sent_tokens[idx]
        return chat_bot_response

# starting and ending phrases
flag=True
print("Chatbot: My name is Chattie. I will answer all your questions about Chatbots. If you want to exit, type Bye!")

while(flag==True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("Chatbot: You are welcome.")
        else:
            if(greeting(user_response)!=None):
                print("Chatbot: " + greeting(user_response))
            else:
                print("Chatbot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Chatbot: Bye! Take care!")


