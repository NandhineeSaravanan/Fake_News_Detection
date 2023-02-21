# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:45:55 2023

@author: sneka
"""

# import library
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression

app = FastAPI()
pickle_in = open("text_classifier.pkl","rb")
text_classifier=pickle.load(pickle_in)



# Define the input data structure
class NewsItem(BaseModel):
    text: str


# Define the prediction function
def predict_news_type(news_item: NewsItem):
    # Clean the news text
    cleaned_text = clean_text(news_item.text)
    
    # Make a prediction using the model
    prediction = text_classifier.predict([cleaned_text])[0]
    is_fake = True if prediction == 1 else False    
    
    return {"The Given Statement is": is_fake}

# Define a function to clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join the lemmatized tokens back into a single string
    cleaned_text = ' '.join(lemmatized_tokens)
    
    return cleaned_text


# Define the prediction endpoint
@app.post("/predict")
async def predict(news_item: NewsItem):
    return predict_news_type(news_item)

   
   
     
    

        
   
