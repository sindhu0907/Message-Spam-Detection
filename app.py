import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()



def transform_text(text):
    text=text.lower()   # converting into lower case
    text=nltk.word_tokenize(text)    # breaking into words ->converted into list
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tfidf= pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Message Spam Detection")
input_sms = st.text_input("Enter the message")


if st.button("Predict"):
    #preprocess
    transform_sms = transform_text(input_sms)
    #vectorize
    vector_input =tfidf.transform([transform_sms])
    #predict
    result= model.predict(vector_input[0])
    #display
    if result ==1:
        st.header("Spam Message")
    else:
        st.header("Not a Spam Message")



