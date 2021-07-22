# Keyword-extraction-through-TF-IDF
import pandas as pd

# read json into a dataframe
df_idf=pd.read_json("data/stackoverflow-data-idf.json",lines=True)

# print schema
print("Schema:\n\n",df_idf.dtypes)
print("Number of questions,columns=",df_idf.shape)

Schema:

accepted_answer_id          float64
answer_count                  int64
body                         object
comment_count                 int64
community_owned_date         object
creation_date                object
favorite_count              float64
id                            int64
last_activity_date           object
last_edit_date               object
last_editor_display_name     object
last_editor_user_id         float64
owner_display_name           object
owner_user_id               float64
post_type_id                  int64
score                         int64
tags                         object
title                        object
view_count                    int64
dtype: object
Number of questions,columns= (20000, 19)


import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))

#show the second 'text' just for fun
df_idf['text'][2]

from sklearn.feature_extraction.text import CountVectorizer
import re

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

#load a set of stop words
stopwords=get_stop_words("resources/stopwords.txt")

#get the text column 
docs=df_idf['text'].tolist()

#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)

cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
word_count_vector=cv.fit_transform(docs)

list(cv.vocabulary_.keys())[:10]


#output
['serializing',
 'private',
 'struct',
 'public',
 'class',
 'contains',
 'properties',
 'string',
 'serialize',
 'attempt']
