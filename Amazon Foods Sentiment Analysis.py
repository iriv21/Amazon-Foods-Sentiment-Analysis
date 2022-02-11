# This program will conduct a Sentiment Analysis. 
# Data is collected and analyzed to gauge customer response through bar graphs,
# wordclouds, .
# Amazon Fine Food Review dataset from Kaggle. 


# Import Amazon file
import pandas as pd
df = pd.read_csv('/Users/irisvargas/Documents/homework/Amazon Fine Foods/Reviews.csv')
# Check file
# print(df.head())

# Import libraries for visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
sns.set_theme(style="darkgrid")


# Using Seaborn as a bar graph to analyze reviews.
fig = px.histogram(df, x="Score")
fig.update_traces(marker_color="lightslategray",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Score')
fig.show()
# Analyzation: This figure shows positive ratings from customers. 


# Create WordClouds
from wordcloud import WordCloud, STOPWORDS 
import nltk
from nltk.corpus import stopwords
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
textt = " ".join(review for review in df.Text)
wordcloud = WordCloud(stopwords=stopwords).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()
### Analyzation:
# When this code runs, we can see popular words expressed from customer reviews.
# The words 'product', followed by 'taste', 'love' and 'Amazon' indicate
# positive sentiments in this dataset.


# Create two data frames with the classified 'positive' and 'negative' reviews.
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda rating : +1 if rating > 3 else -1)
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]


# Cleaning Data words to enhance a better discriptive wordcloud.
from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS)
stopwords.update(["br", "href","good","great","better","taste"]) 


# Positive sentiment Wordcloud
pos = " ".join(review for review in positive.Summary)
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show() 


# Negative sentiment Wordcloud
# I noticed the negative wordcloud resemebled the positive wordcloud.
# By removing the words 'great', 'good' and 'better', the negative wordcloud
# had a better visualization and summary of negative words. 
neg = " ".join([str(review) for review in negative.Summary])
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud33.png')
plt.show()


# Creating a data model to represent the difference of negative vs positive reviews.
df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(df, x="sentimentt")
fig.update_traces(marker_color="steelblue",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


#### Building the Sentiment Analysis Model

# This function will clean the data to predict the outcome of reviews
# by user input.
# First remove punctuation 
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final
df['Text'] = df['Text'].apply(remove_punctuation)
df = df.dropna(subset=['Summary'])
df['Summary'] = df['Summary'].apply(remove_punctuation)

# Second, splitting the new data frame into two columns
# "Summary" and "Sentiment" to predict the outcome.
dfNew = df[['Summary','sentiment']]


index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]


# Import library for vectorizing, this is for future use to model
# our logistic regression model.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# Cleaning data further by splitting target and independent variables.
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


lr.fit(X_train,y_train)
predictions = lr.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)
# Finally, print the report
print(classification_report(predictions,y_test))

# All these visualization models are to represent customer reviews.
# The creation of models represents all positive, negative and neutral 
# feedback by using a histogram model. The histogram model displayed scores 1-5. 
# The wordclouds were to represent an iteration of the most used words customers 
# were writing both in negative and positive feedback.
# The final code is the report that suggests the future
# reviews of customers based on the classification report.