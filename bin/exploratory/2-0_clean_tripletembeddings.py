import pandas as pd

'''
## Read in data files
'''
#Create empty dic
df_scores = {}

#The n decide which file to be read in
for n in range(2, 21):
    df_scores[n] = pd.read_csv(r'..\data\process\tste\tste_embedding_' + str(n) + '.csv', encoding='utf-8', header=None)


'''
## Concat all scores in one df
'''
#Concat all together
#But this results in a hierarchical index
df_score = pd.concat(df_scores, axis=1)

#Resolve the hierarchical index
#1. Transform the indices into str
#2. Join the hierarchical indices
#3. Join 'tste' with the joined indices 
df_score.columns = ['_'.join(['tste','_'.join([str(i) for i in indices])]) for indices in df_score.columns.values]


'''
## Save the result as csv
'''
df_score.to_csv(r'..\data\process\tste\tste_concat.csv')