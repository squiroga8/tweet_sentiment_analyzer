"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

## import libraries
import keras
import tensorflow as tf
from keras import backend as K 
from tensorflow import Graph, Session

from keras.models import model_from_json
import numpy as np 
import pickle
from keras.preprocessing.sequence import pad_sequences

## load model and tokenizer
with open('mood_tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open('mood_model.json', 'r') as file:
    model_json = file.read()

## tensor flow stuff to load things on the backend, for loading a model that will
## need to be accessible from any part of the app. if more than one model, copy and 
## paste and change to graph2
global model 
graph1 = Graph()
with graph1.as_default():
    session1 = Session(graph=graph1)
    with session1.as_default():
        model = model_from_json(model_json)
        model.load_weights('mood_model.h5')

###
# Routing for your application.
###

@app.route('/')
def home():
    """Render website's home page."""
    return render_template('home.html', mood="", tweet_exhibits_text="")

@app.route('/analyze_tweet', methods=['POST'])
def analyze_tweet():
    tweet = [str(request.form['tweet'])]
    labels = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
    encoded_tweet = tokenizer.texts_to_sequences(tweet)
    padded_tweet = pad_sequences(encoded_tweet, maxlen=256)

    K.set_session(session1)
    with graph1.as_default():
        preds = model.predict_proba(padded_tweet)
    mood = labels[np.argmax(preds)]
    return render_template('home.html', mood=mood, tweet_exhibits_text="This tweet exhibits:")

## @app.route('/about/')
## def about():
   ## """Render the website's about page."""
 ##   return render_template('about.html')


##@app.route('/<file_name>.txt')
##def send_text_file(file_name):
##    """Send your static text file."""
##    file_dot_text = file_name + '.txt'
##    return app.send_static_file(file_dot_text)


###
# The functions below should be applicable to all Flask apps.
###

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=600'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
