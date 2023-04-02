
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from termcolor import colored
import json
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# nltk.download('stopwords')
# nltk.download('punkt')

file_name = 'Sarcasm_Headlines_Dataset.json'
data = []
with open(file_name) as f:
    for line in f:
        data.append(json.loads(line))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

headlines = [item['headline'] for item in data]

preprocessed_headlines = []
for headline in headlines:
    preprocessed_headline = preprocess_text(headline)
    preprocessed_headlines.append(preprocessed_headline)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_headlines).toarray()
y = [item['is_sarcastic'] for item in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, np.array(y_train), epochs=20, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

def is_sarcastic(headline):
    preprocessed_headline = preprocess_text(headline)
    features = vectorizer.transform([preprocessed_headline]).toarray()
    prediction = (model.predict(features) > 0.5).astype("int32")
    return prediction[0][0] == 1

headline = "Nation's Dogs Vow To Bark Relentlessly At Nothing In Particular"
print(is_sarcastic(headline))

while(headline != "exit"):
    headline = input("Enter a Headline!: ")
    message = headline + " is a Sarcastic Headline!\n"

    if (is_sarcastic(headline)):
        message = headline + " is not a Sarcastic Headline!\n"
        print(colored(message, 'green'))
    else:
        print(colored(message, 'red'))
    