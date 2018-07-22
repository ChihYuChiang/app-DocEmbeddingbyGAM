#%%
import pandas as pd
import pickle

import nltk
import gensim


#%%
#--Read in data
df = pickle.load(open(r'..\data\process\df_normalized.p', 'rb'))
print(len(df))


#--Switch
W2V = False
D2V = True




'''
-----------------------------------------------------------
Perform W2V embeddings
-----------------------------------------------------------
'''
if W2V:
#%%
    #--Gensim implementation of Word2Vec -- with stopwords
    W2V_Wstop = gensim.models.word2vec.Word2Vec(df['Review_normalized_sent_Wstop'].sum(), workers=6)
    W2V_Wstop.save(r'..\data\process\W2V_Wstop')


#%%
    #--Gensim implementation of Word2Vec -- without stopwords
    W2V_WOstop = gensim.models.word2vec.Word2Vec(df['Review_normalized_sent_WOstop'].sum(), workers=6)
    W2V_WOstop.save(r'..\data\process\W2V_WOstop')




'''
-----------------------------------------------------------
Perform D2V embeddings
-----------------------------------------------------------
'''
if D2V:
    #%%
    #--Tag docs by game title
    taggedDocs = []
    for index, row in df.iterrows():
        taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words=row['Review_normalized_arti_WOstop'], tags=[row['Game'], 'id_' + str(index)]))
    df['TaggedReview'] = taggedDocs


    #%%
    #--Gensim implementation of Doc2Vec -- with stopwords
    D2V_WOstop = gensim.models.doc2vec.Doc2Vec(df['TaggedReview'], size=300, workers=6) #Limiting to 300 dimensions
    D2V_WOstop.save(r'..\data\process\D2V_WOstop')


    #%%
    #--Tag docs by game title
    taggedDocs = []
    for index, row in df.iterrows():
        taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words=row['Review_normalized_arti_Wstop'], tags=[row['Game'], 'id_' + str(index)]))
    df['TaggedReview'] = taggedDocs


    #%%
    #--Gensim implementation of Doc2Vec -- without stopwords
    D2V_Wstop = gensim.models.doc2vec.Doc2Vec(df['TaggedReview'], size=300, workers=6) #Limiting to 300 dimensions
    D2V_Wstop.save(r'..\data\process\D2V_Wstop')
    D2V_Wstop.docvecs.doctags

