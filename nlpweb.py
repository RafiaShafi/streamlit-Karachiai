#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:16:18 2022

@author: gaditek
"""
import nltk
import streamlit as st
import numpy as np
import pandas as pd
import neattext.functions as nfx
import matplotlib.pyplot as plt 
import seaborn as sns 
#from wordcloud import WordCloud 
from PIL import Image
nltk.download('stopwords')
from nltk.corpus import stopwords
image = Image.open('/home/gaditek/Downloads/Jupyter/NLP.png')
st.image(image, caption = 'Natural language processing', use_column_width = True)
pd.options.mode.chained_assignment = None
st.title('Natural language processing in SMS_Data')
st.markdown('Done By Rafia.sk')
st.markdown('Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written -- referred to as natural language. It is a component of artificial intelligence (AI). NLP has existed for more than 50 years and has roots in the field of linguistics.')

df = pd.read_csv('/home/gaditek/Downloads/Jupyter/SMS_data.csv', encoding= 'unicode_escape')
df.columns = ['Sno','Date','Message_body', 'Label']



df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month_name())




df["Label"].value_counts()


#userhandles
df["Mas_new"] = df["Message_body"].apply(nfx.remove_userhandles)
#stopword
df["Mas_new"] = df["Message_body"].apply(nfx.remove_stopwords)
df["Mas_new"] = df["Message_body"].apply(nfx.remove_special_characters)
df["Msg_lower"] = df["Mas_new"].str.lower()

nltk.download('WordNetLemmatizer')
#nltk.download('wordnetlemmatizer')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(Msg_lower):
    return " ".join([lemmatizer.lemmatize(word) for word in Msg_lower.split()])
df["Msg_lemmatized"] = df["Msg_lower"].apply(lambda Msg_lower: lemmatize_words(Msg_lower))

st.header("Processed Data")
st.dataframe(df)
button1 = st.button('Aanlyz spam and Non-Spam at a Glance')
if button1:
   fig = plt.figure()
   df["Label"].value_counts().plot(kind = "bar")
   st.pyplot(fig)
   
from collections import Counter
cnt = Counter()
for text in df["Msg_lemmatized"].values:
    for word in text.split():
        cnt[word] += 1
        
### Word Cloud of mostly used word in our Group


df2 = df.query("Label == 'Spam'")
df3 = df.query("Label == 'Non-Spam'")

from collections import Counter
cnt = Counter()
for text in df2["Msg_lemmatized"].values:
    for word in text.split():
        cnt[word] += 1
rs = cnt.most_common(20)



spm = pd.DataFrame(rs, columns=['Word','Count'])

from collections import Counter
cnt = Counter()
for text in df3["Msg_lemmatized"].values:
    for word in text.split():
        cnt[word] += 1
rsnon = cnt.most_common(20)
spmnon = pd.DataFrame(rsnon, columns=['Word','Count'])

#fig1 = plt.figure()
#st.bar_chart(spm[:100])
#st.pyplot(fig1)  
#st.bar_chart(spm['Word'])

#def spam_data(msgtype):
  #  st.markdown("List of Spam Messages")
 #   df2 = df.query("Label == 'Spam'")
   # st.dataframe(df2)
   # if msgtype == 'Spam"
  #     result = 
 # if msgtype == 'Spam':
 #  show = pd.DataFrame(cnt.most_common(10),columns=['Word','Count']).plot(kind='barh',x='Word',y='Count')
    #     cnt.most_common(10)
     
    #else:
     #    show = 'Hello,' + 'Mr.' + name.capitalize()
    #return show

def main():
    st.header('Choose Message type for Viusalization')
    msgtype = st.selectbox('Select here', ['Select ', 'Spam', 'Non-Spam'])
    if msgtype == "Spam":
        b1,b2 =st.beta_columns(2)
        with b1:
            st.line_chart(spm['Word'])
        with b2:
            st.bar_chart(spm['Word'])

    else:
        c1,c2 = st.beta_columns(2)
        with c1:
            st.line_chart(spmnon['Word'])
            #st.dataframe(df3)
        with c2:
            st.bar_chart(spmnon['Word'])
     #   show = spam_data(msgtype)
      #  st.success(show)
        
#st.snow()


        
if __name__ == '__main__':
    main()




