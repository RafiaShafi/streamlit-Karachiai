#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:51:39 2022

@author: gaditek
"""


# read airpassenger.csv into python
# create a dropdown for year hint: st.selectbox() ; boolen mask
# show table of values for that year obly # st.table(df)
import streamlit as st


import pandas as pd
df = pd. read_csv ('AirPassengers.csv')
print(df)

df.columns = ['Date','Number of Passengers']



df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month_name())
df.head()

print(df)


def main():
    st.title('Data of the Month')
    #Syear = st.text_input('year')
    #print(year)
    year = st.selectbox('select year', df['Year'].unique())
    
  #  gender = st.selectbox('Year', ['1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959', '1960'])
  
    b = st.button('show data')
    if b:
      subset = df[df['Year'] == year]
      st.table(subset)
    
    
if __name__ == '__main__':
    main()
    
