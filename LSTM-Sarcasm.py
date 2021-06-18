import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

#Open the dataset (is a json file)
with open(r'C:\Users\rodri\OneDrive\Escritorio\Tec\Sarcasm text\Sarcasm_Headlines_Dataset_v2.json', 'r') as f:
    datastore = json.load(f)

Sentences = []
labels = []
urls = []

#Divide the data into categories
for item in datastore:
    Sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

#divide the data into training and testing
training_size = 20000
training_sentences = Sentences[0:training_size]
testing_sentences = Sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

#Set the values and declare the tokenizer
vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

#We start modeling the model, and then we need to compile it and fit it
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(48, activation = 'relu'),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
history = model.fit(training_padded, training_labels, epochs = num_epochs, validation_data = (testing_padded, testing_labels), verbose = 2)


sentence = ["mom starting to fear son's web series closest thing she will have to grandchild"]
sentence = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sentence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))