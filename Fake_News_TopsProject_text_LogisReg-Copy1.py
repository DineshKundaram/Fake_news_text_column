import pandas as pd 
import numpy as np 
import re 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder

# downloading stopwords from nltk library
import nltk
nltk.download('stopwords')
print(stopwords.words('english'))
#reading csv data

news = pd.read_csv('news.csv')
#creating input and output variable

train = news['text']
test = news['label']
# now target column has string values so need to encode into interger
le = LabelEncoder()
test = le.fit_transform(test)
#creating PorterStemmer variable
port_stem = PorterStemmer()

# applying stemming Process
def stemming(content): #defining function
    
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) 
    #here applying regular expression substitute, '^' means excluding a to z characters, replacing with '_'(space) in content 
    
    stemmed_content = stemmed_content.lower()
    #apply string values in lower case
    
    stemmed_content = stemmed_content.split() 
    #splitting strings
    
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    #applying PorterStemmer algorithm on stemmed_content which are not in stopwords library
    
    stemmed_content =' '.join(stemmed_content)
    #joining all words again into same sentences
    
    return stemmed_content



#applying stemming on train dataset
train = train.apply(stemming)
vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(train)
x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.2, stratify=test, random_state=2)

model = LogisticRegression()
model.fit(x_train, y_train)

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('fake_news_text.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = model.predict(x_test)
    return render_template('index.html')

if __name__ =='__main__':
    app.run(debug=True)