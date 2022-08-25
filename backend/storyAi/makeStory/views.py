from __future__ import absolute_import, division, print_function, unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
import os, shutil
import tensorflow as tf
import keras
from keras.models import load_model
import keras.models
import os, numpy, sys, time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Create your views here.
# Vonnegut vocab size: 61
# Dick vocab size: 54

class WriterMind(object):
    
    def read_vocab(self):
        with open(self) as file:
            chars = file.read().split(" ")
            chars.insert(1, " ")
        return chars

    def vocab_processing(self):
        char2idx = {u:i for i, u in enumerate(vocab)}
        idx2char = np.array(self)

        return char2idx, idx2char

    def build_model(self, embedding_dim, rnn_units, batch_size):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    self, embedding_dim, batch_input_shape=[batch_size, None]
                ),
                tf.keras.layers.GRU(
                    rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform',
                ),
                tf.keras.layers.Dense(self),
            ]
        )

    def generate_text(self, start_string, num_generate, char2idx, idx2char):
    		
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []
        temperature = 1.0

        self.reset_states()
        for _ in range(num_generate):
            predictions = self(input_eval)

            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + '\n' + ''.join(text_generated))

class WriterAnswer(object):

    def create_story(self):

        embedding_dim = 256
        rnn_units = 1048
        katalog = "AI_weights"
        author = self['author']
        text = self['storyBegin']
        vocab_path = f"{os.getcwd()}/{katalog}/{author}/vocab.txt"
        weights_path = f"{os.getcwd()}/{katalog}/{author}/ckpt_100"

        tf.compat.v1.enable_eager_execution()

        chars = WriterMind.read_vocab(vocab_path)
        vocab_size = len(chars)

        model = WriterMind.build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        char2idx, idx2char = WriterMind.vocab_processing(chars)

        model.load_weights(weights_path)

        self = WriterMind.generate_text(model, text[-100:], 600, char2idx, idx2char)

        return self
    