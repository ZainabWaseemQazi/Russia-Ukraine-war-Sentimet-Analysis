import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("stopwords")

#Read file
data=pd.read_csv("filename.csv")
print(data.head())
#columns
print(data.columns)
#we only take username, tweet, language columns for analysis
data=data[["username", "tweet", "language"]]
#three columns to use
print(data.head())
#checking for null values in cloumns
print(data.isnull().sum())
#checking number to tweets in each language
print(data['language'].value_counts())

#preprocessing data for further analysis
stemmer=nltk.SnowballStemmer("english")
stopword=set(stopwords.words("english"))

def cleantext(text):
    text=str(text).lower()
    text=re.sub('https?://\S+|www\.\S+', " ", text)                                 #removing links
    text=re.sub("[^\w\s]", " ", text)                                               #removing punctutation or extracting words/strings
    text=re.sub("\n", " ", text)                                                    #removing nextline
    text=re.sub("\w*\d\w*", " ", text)                                              #removing digits
    text = [word for word in text.split(" ") if word.casefold() not in stopword]    #removing stopwords
    text = " ".join(text)                                                           #joining all words
    text=[stemmer.stem(word) for word in text.split(" ")]                           #stemming of text
    text=" ".join(text)                                                             #joining the text
    return text
data["tweet"]=data["tweet"].apply(cleantext)
print(data["tweet"])
print('Data is cleaned')
#creating wordcloud of tweets to see most frequently words
text=" ".join((i for i in data.tweet))
wordcloud=WordCloud(background_color="white").generate(text)
plt.figure(figsize=(14, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#adding three more attributes to check the polarity of tweets
nltk.download("vader_lexicon")
sentiments=SentimentIntensityAnalyzer()
data["Positive"]=[sentiments.polarity_scores(i) ["pos"] for i in data["tweet"]]
data["Negative"]=[sentiments.polarity_scores(i) ['neg'] for i in data["tweet"]]
data["Neutral"]=[sentiments.polarity_scores(i) ['neu'] for i in data ["tweet"]]
data=data[["tweet", "Positive", "Negative", "Neutral"]]
print(data.head())

#Checking frequent positive tweet words
positive=' '.join([i for i in data["tweet"] [data["Positive"]>data["Negative"]]])
wordcloud=WordCloud(background_color="white").generate(positive)
plt.figure(figsize=(14,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#checking frequent negative tweet words
negative=' '.join([i for i in data["tweet"] [data["Positive"]<data["Negative"]]])
wordcloud=WordCloud(background_color="white").generate(negative)
plt.figure(figsize=(14,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



