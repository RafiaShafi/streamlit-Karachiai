#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:16:18 2022

@author: gaditek
"""

import streamlit as st
import numpy as np
import pandas as pd
import nltk
nltk.download('all')
import neattext.functions as nfx
import matplotlib.pyplot as plt 
#from wordcloud import WordCloud 
from PIL import Image
nltk.download('stopwords')
from nltk.corpus import stopwords

#iamge show
image = Image.open('NLP.png')

#header
st.image(image, caption = 'Natural language processing', use_column_width = True)
pd.options.mode.chained_assignment = None
st.title('Natural language processing in SMS_Data')
st.markdown('Done By Rafia.sk')
st.markdown('Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written -- referred to as natural language. It is a component of artificial intelligence (AI). NLP has existed for more than 50 years and has roots in the field of linguistics.')


#file import
df = pd.read_csv('SMS_data.csv', encoding= 'unicode_escape')
#rebane coloumn
df.columns = ['Sno','Date','Message_body', 'Label']


#data setting accrding to date and month
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month_name())


#df["Label"].value_counts()

#userhandles
df["Mas_new"] = df["Message_body"].apply(nfx.remove_userhandles)
#stopword
df["Mas_new"] = df["Message_body"].apply(nfx.remove_stopwords)
#special characters
df["Mas_new"] = df["Message_body"].apply(nfx.remove_special_characters)
#lowercase msgs
df["Msg_lower"] = df["Mas_new"].str.lower()

nltk.download('WordNetLemmatizer')
#nltk.download('wordnetlemmatizer')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(Msg_lower):
    return " ".join([lemmatizer.lemmatize(word) for word in Msg_lower.split()])
df["Msg_lemmatized"] = df["Msg_lower"].apply(lambda Msg_lower: lemmatize_words(Msg_lower))

st.header("Processed Data")

#data show
st.dataframe(df)

button1 = st.button('Aanlyz spam and Non-Spam at a Glance')
if button1:
   fig = plt.figure()
   df["Label"].value_counts().plot(kind = "bar")
   st.pyplot(fig)
   

### Word Cloud of mostly used word in our Group

df2 = df.query("Label == 'Spam'")
df3 = df.query("Label == 'Non-Spam'")


#msg count on dates

nmsg = df3.groupby('Date')['Msg_lower'].count().sort_values(ascending = False )
nmasd = pd.DataFrame(nmsg)
#st.dataframe(nmasd)


smsg = df2.groupby('Date')['Msg_lower'].count().sort_values(ascending = False )
smasd = pd.DataFrame(smsg)
#st.dataframe(smasd)

freq = pd.Series(' '.join(df2['Msg_lemmatized']).split()).value_counts()[:20]

#st.bar_chart(freq)

freqns = pd.Series(' '.join(df3['Msg_lemmatized']).split()).value_counts()[:20]

def main():
    st.header('Choose Message type for Viusalization')
    msgtype = st.selectbox('Select here', ['Select ', 'Spam', 'Non-Spam'])
    if msgtype == "Spam":
        b1,b2 =st.beta_columns(2)
        with b1:
            st.line_chart(smasd['Msg_lower'])
        with b2:
            st.bar_chart(freq)

    else:
        c1,c2 = st.beta_columns(2)
        with c1:
            st.line_chart(nmasd['Msg_lower'])

            #st.dataframe(df3)
        with c2:
            st.bar_chart(freqns)
     #   show = spam_data(msgtype)
      #  st.success(show)
        
#st.snow()


        
if __name__ == '__main__':
    main()


#st.bar_chart(freqns)

#df3 = px.data.tips()
#fig3 = px.bar(df3)
#fig3.show()
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





