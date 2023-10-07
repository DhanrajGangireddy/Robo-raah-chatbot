import nltk
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
clss = []
docmts = []
igwords = ['/', '!', ',', '?', '.','>','<']

data_file = open('intents2.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)  # Tokenization
        words.extend(w)
        docmts.append((w, intent['tag']))  # Append to docmts

        if intent['tag'] not in clss:  # clss list
            clss.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in igwords]  # Lemmatize and remove duplicates
words = sorted(list(set(words)))  # Words arranged in alphabetical order
clss = sorted(list(set(clss)))

print(len(docmts), "docmts")
print(len(clss), "class", clss)
print(len(words), "unique words", words)

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(clss, open('labels.pkl', 'wb'))

# Training data 
training = []
output_empty = [0] * len(clss)  # Empty array for output
for doc in docmts:
    bag = []
    pwords = doc[0]
    pwords = [lemmatizer.lemmatize(word.lower()) for word in pwords]
    for w in words:
        bag.append(1) if w in pwords else bag.append(0)

    row_o = list(output_empty)
    row_o[clss.index(doc[1])] = 1

    training.append([bag, row_o])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Data created for training")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=24, verbose=1)
model.save('model.h5')

print("Model created")
