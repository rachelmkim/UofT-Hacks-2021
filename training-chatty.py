"""UofTHacks"""

import tensorflow as tf
import clean_data

import numpy as np
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer

NUM_EPOCHS = 80
BATCH_SIZE = 5

lemmatizer = WordNetLemmatizer()

words = []
categories = []
docs = []

questions = clean_data.clean_questions
answers = clean_data.clean_answers
unnecessary = ['?', '!', '.']

for i in range(len(questions)):
    statement = answers[i].split(' ')
    for word in statement:
        # tokenize word
        w = nltk.word_tokenize(word)
        words.extend(w)
        # adding documents
        docs.append((w, questions[i]))

        # add question to categories, if not already included
        if questions[i] not in categories:
            categories.append(questions[i])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in unnecessary]
words = sorted(list(set(words)))

classes = sorted(list(set(categories)))

# Initialize empty list meant to store training data
training = []
output_empty = [] * len(classes)

# From here on, it is the standard method to train a model
for doc in docs:
    # Initialize empty bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# randomly rearrange training data
random.shuffle(training)

# Convert to np.array, which helps in specifying shape of hidden layers
training = np.array(training)

# Constructing training and testing lists.
x_train = list(training[:, 0])
y_train = list(training[:, 1])
print("Training data created")

# Create model with 3 layers.
# First and second layers have 128 neurons.
# 3rd layer is output, and contains as many neurons as there are answers
# Reply predicted using softmax
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
# model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(len(y_train[0]), activation='softmax'))

# Compile model.
# For this chatbot, we choose stochastic gradient descent
stochastic = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=stochastic, loss='categorical_crossentropy', metrics=['accuracy'])

# Model fitting
hist = model.fit(np.array(x_train), np.array(y_train), epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE, verbose=1)
# Note: Make verbose=0 after verifying program works
model.save('chatbot_model.h5', hist)

print("model created")
