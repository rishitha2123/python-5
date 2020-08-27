#!/usr/bin/env python
# coding: utf-8

# In[72]:


### TEXT GENERATION PROJECT


# In[73]:


# importing dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


# In[41]:


# load data
# loading data and opening our input data in the form of a txt file
# project Gutenburg/berg is where the data can be found
file = open("frankenstein[1].txt").read()


# In[42]:


# tokenization
# standardization
# tokenization is a process of breaking a stream of text up into words phrases symbols or other meaningful elements

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)


# In[43]:


# preprocess the input data, make tokens
processed_inputs = tokenize_words(file)


# In[64]:


# chars to numbers
# convert characters to our input numbers 
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i, c in enumerate(chars))


# In[45]:


# check if words to chars or chars to num (?!) has worked ?
input_len = len(processed_inputs)
vocab_len = len(chars)
print("total number of characters: ", input_len)
print("total vocab: ", vocab_len)


# In[46]:


# seq length
seq_length = 100
x_data = []
y_data = []


# In[47]:


# loop through the sequence
for i in range(0, input_len - seq_length, 1):    
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)


# In[48]:


# convert input sequence to np array and so on
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)


# In[53]:


# one-hot encoding
y = np_utils.to_categorical(y_data)


# In[60]:


# creating the model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[61]:


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[62]:


# saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]


# In[63]:


# fit model and let it train
model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)


# In[65]:


# recompile model with the saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[66]:


# output of the model back into characters
num_to_char = dict((i, c) for i, c in enumerate(chars))


# In[67]:


# random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


# In[68]:


# generate the text 
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]


# In[70]:


### DONE!!!


# In[71]:


### THANKYOU !!!

