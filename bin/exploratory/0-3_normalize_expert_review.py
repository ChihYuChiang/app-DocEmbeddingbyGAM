#%%
import pandas as pd
import pickle

import nltk
import numpy as np
import gensim
import re

#Read in data
df = pd.read_csv(r'..\data\df_cb_main_combined.csv', index_col=0, encoding='utf-8', error_bad_lines=False).dropna(subset=['Review']).drop_duplicates(['Author', 'Game'])




'''
------------------------------------------------------------
Normalization Settings
------------------------------------------------------------
'''
#%%
#--Function for normalization
def normlizeTokens(tokenLst, stopwordLst=None, stemmer=None, lemmer=None, Xvocab=None, Ovocab=None):

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Lemmertize
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #Stem
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)
    
    #Remove unwanted words by reg
    if Xvocab is not None:
        Xvocab_str = '|'.join(Xvocab)
        workingIter = (w for w in workingIter if not re.match(Xvocab_str, w))

    #Include ONLY wanted words by reg
    if Ovocab is not None:
        Ovocab_str = '|'.join(np.array(Ovocab))
        workingIter = (w for w in workingIter if re.fullmatch(Ovocab_str, w))

    #Remove stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
    return list(workingIter)


#--Initialize the tools
stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()
badvocab = []
expKeywords = pd.read_csv(r'..\data\output\keywordGroup_hierarchy_10.csv').keyword #Exp keyword list


#--Test the function
testPhrases = ['At', 'first', 'glance', '05', 'glances', 'firsts', '1', "can't", 'hi', '-', 'a-ha']
normlizeTokens(testPhrases, stopwordLst=None, stemmer=None, lemmer=None, Xvocab=None, Ovocab=expKeywords)




'''
------------------------------------------------------------
Title-Removing Settings
------------------------------------------------------------
'''
#%%
#--Create a list of game title phrases to be removed
#Define processing function
def titleWords(titles):

    #Remove parenthesis and content inside eg. (1991)
    titles = [re.sub('\(.+\)|\[.+\]', '', title).strip() for title in titles]

    #Remove number and number with "," separation
    wkTitles = [re.sub('[0-9]+,?[0-9]+', '', title) for title in titles]

    #Split title into phrases by ":" and " - " (reviews use part of the title to refer to the game eg. Resident Evil: Crazy)
    wkTitles = [re.split(':| - ', title) for title in wkTitles]

    #Remove straying spaces at the beginning and end
    wkTitles = [title.strip() for titles in wkTitles for title in titles]

    #Also include the original titles (full length, only remove parenthesis, retain numbers)
    #Also include the capitalized version of some all-capitalized on-word titles eg. STRAFE -> Strafe
    wkTitles = wkTitles \
        + list(titles) \ 
        + [title.capitalize() for title in titles if not re.search(' ', title)]
    
    #Remove duplicate phrase
    wkTitles = list(set(wkTitles))

    #Remove cumbersome words by a filter function
    wkTitles = list(filter(filterWord, [re.escape(title) for title in wkTitles]))

    #Sort to make the longest phrase at the head of the list. They will be removed first when plugged in
    wkTitles = sorted(wkTitles, key=len, reverse=True)

    return wkTitles

#Filter function for cumbersome phrases
def filterWord(word):
    if word == '': return False
    if len(word) == 1: return False
    if re.match('[a-z]', word): return False
    return True


#--Produce the reg pattern
#(title1|title2)(.| |'s)
titleSubStr = '(' + '|'.join(titleWords(df.Game)) + ")(.| |'s)"




'''
------------------------------------------------------------
Implementation for tokenized and normalized words
------------------------------------------------------------
'''
#--Logical gate
useSentence = False
useArticle  = True


#%%
if useSentence:
    #--Sentence based operations
    #Tokenize
    df['Review_tokenized_sent'] = df['Review'].astype('str').apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])

    #Normalization
    df['Review_normalized_sent_Wstop'] = df['Review_tokenized_sent'].apply(lambda x: [normlizeTokens(s, stopwordLst=None, stemmer=None, lemmer=None, Xvocab=None, Ovocab=None) for s in x])
    df['Review_normalized_sent_WOstop'] = df['Review_tokenized_sent'].apply(lambda x: [normlizeTokens(s, stopwordLst=stop_words_nltk, stemmer=None, lemmer=None, Xvocab=None) for s in x])


#%%
if useArticle:
    #--Article based operations
    #Tokenize and remove game titles from review text (case sensitive)
    df['Review_tokenized_arti'] = df['Review'].astype('str').apply(lambda x: nltk.word_tokenize(re.sub(titleSubStr, '', x)))

    #Normalization
    df['Review_normalized_arti_Wstop'] = df['Review_tokenized_arti'].apply(lambda x: normlizeTokens(x, stopwordLst=None, stemmer=None, lemmer=wordnet, Xvocab=None, Ovocab=expKeywords))
    df['Review_normalized_arti_WOstop'] = df['Review_tokenized_arti'].apply(lambda x: normlizeTokens(x, stopwordLst=stop_words_nltk, stemmer=None, lemmer=wordnet, Xvocab=None, Ovocab=expKeywords))


#%%
#--Save all results (a fat file)
pickle.dump(df, open(r'..\data\process\df_normalized.p', 'wb'))
