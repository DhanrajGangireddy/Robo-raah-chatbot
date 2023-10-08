from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras.models import load_model

app = Flask(__name__)

nltk.download('punkt')
nltk.download('wordnet')

model = load_model('model.h5')
words = pickle.load(open('texts.pkl', 'rb'))
clss = pickle.load(open('labels.pkl', 'rb'))
with open('intents2.json', 'r') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

def sentencecln(sentence):
    wordsen = nltk.word_tokenize(sentence)
    wordsen = [lemmatizer.lemmatize(word.lower()) for word in wordsen]
    return wordsen

def group(sentence, words, details=True):
    wordsen = sentencecln(sentence)
    bag = [0]*len(words)
    for s in wordsen:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    usermsg = request.form['usermsg']
    p = group(usermsg, words)
    res = model.predict(np.array([p]))[0]
    Err_threshold = 0.25
    fil_results = [[i, r] for i, r in enumerate(res) if r > Err_threshold]
    fil_results.sort(key=lambda x: x[1], reverse=True)
    
    Predict_result = clss[fil_results[0][0]]
    
    response = None
    for intent in intents['intents']:
        if intent['tag'] == Predict_result:
            response = random.choice(intent['responses'])
            break
    
    if response:
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'Im sorry, I couldnt understand your question.'})


if __name__ == '__main__':
    app.run(debug=True)
