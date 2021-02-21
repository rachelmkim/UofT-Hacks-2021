"""
UofTHacks 2021 Project Chatty
Module for generating a response
"""

import numpy as np
from keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer

from typing import List, Dict
from clean_data import questions as questions

import pickle
import random

lemmatizer = WordNetLemmatizer()

ERROR_MARGIN = 0.25

model = load_model('chatty.h5')

# intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
categories = pickle.load(open('classes.pkl', 'rb'))


def clean_sentence(sentence: str) -> List[str]:
    """Break down the sentence into using lemmatized words, so that responses may be similar"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str, detail: bool = True) -> np.ndarray:
    """Return whether the words in the string exist in the bag of words, from the training data.

    In the returned array: a 1 is entered for each word in the bag that exists in the sentence,
    and is 0 otherwise.
    """
    # tokenize the pattern
    sentence_words = clean_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if detail:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_category(sentence: str) -> List[Dict]:
    """Predict which category the sentence might fit into.

    Returns a list of dictionaries, where each dictionary has two keys 'question' and
    'probability of question', which function as their names suggest.
    """
    # filter out predictions below a set margin (default set to 0.25)
    p = bag_of_words(sentence, detail=False)
    res = model.predict(np.array([p]))[0]

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_MARGIN]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        dictionary = {"question": categories[result[0]],
                      "probability of question": str(result[1])}
        return_list.append(dictionary)
    return return_list


def respond(message: str) -> str:
    """Return Chatty's response to a given message"""
    question = predict_category(message)
    response = generate_response(question, questions)
    return response


def generate_response(ints, intents_json) -> str:
    """Generate a response"""
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Sorry, I didn't quite catch that"

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
