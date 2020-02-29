from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, ReLU, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.utils import plot_model
import re


class Translator:
    PUNCTUATION = '([' + '!"#$%&()*,+-./:;<=>?@[\\]^_`{|}~' + '])'

    def __init__(self, language_seq_one, language_seq_two):
        self.seq_one = language_seq_one
        self.seq_two = language_seq_two
        self.filter_sequences()
        print(self.seq_one, self.seq_two)
        self.tokenizer_one = self.create_tokenizer(self.seq_one)
        self.tokenizer_two = self.create_tokenizer(self.seq_two)
        self.vocab_size_one = len(self.tokenizer_one.word_index)
        self.vocab_size_two = len(self.tokenizer_two.word_index)
        self.word_vector = np.zeros(self.vocab_size_one)
        self.embedding_length = 128
        self.latent_dim = 256
        self.model = None

        # print(self.tokenizer.texts_to_sequences(self.sequences))

    def create_tokenizer(self, sequences):
        t = Tokenizer(filters="\t\n")
        t.fit_on_texts(sequences)
        return t

    def space_punctuation(self, s):
        """ Pads a single space around all punctuation"""
        s = re.sub(PUNCTUATION, r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        return s

    def filter_sequences(self):
        for i, s in enumerate(self.seq_one):
            s = s.lower()
            s = self.space_punctuation(s)
            self.seq_one[i] = s

        for i, s in enumerate(self.seq_two):
            s = s.lower()
            s = space_punctuation(s)
            self.seq_two[i] = s

    def word_sequences_to_arrays(self, sequences):
        """
        First axis of each training example
        Second axis is of each word in a training example
        Third axis is the word vector
        """
        sequences = self.tokenizer.texts_to_sequences(sequences)

        for i, sequence in enumerate(sequences):
            for j, word_index in enumerate(sequence):
                w_vec = self.word_vector.copy()
                w_vec[word_index - 1] = 1
                sequences[i][j] = w_vec
            sequences[i] = np.array(sequences[i])

        return np.array(sequences)

    def initialize_model(self):

        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(self.vocab_size_one + 1, self.embedding_length, mask_zero=True)
        encoder_embedding_outputs = encoder_embedding(encoder_inputs)
        encoder_LSTM = LSTM(self.latent_dim, return_state=True)
        encoder_LSTM_output, encoder_hidden_state, encoder_cell_state = encoder_LSTM(
            encoder_embedding_outputs)  # The hidden states represent the sentence!

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(self.vocab_size_two + 1, self.embedding_length, mask_zero=True)
        decoder_embedding_outputs = decoder_embedding(decoder_inputs)
        decoder_LSTM = LSTM(self.latent_dim, return_sequences=True, return_state=False)
        decoder_LSTM_output = decoder_LSTM(decoder_embedding_outputs,
                                           initial_state=[encoder_hidden_state, encoder_cell_state])
        decoder_dense = Dense(self.vocab_size_two, activation="softmax")
        decoder_dense_outputs = decoder_dense(decoder_LSTM_output)
        model = Model([encoder_inputs, decoder_inputs], decoder_dense_outputs)
        plot_model(model, show_shapes=True)
        model.compile(loss='categorical_crossentropy')

        self.model = model
        return model


t = Translator(["Hello, how are you sir i hope you are having a mighty fine day"], ["Hola, cómo está, señor Espero que esté teniendo un buen día"])
t.initialize_model()
inp1 = [[1,2,3]]
inp2 = [[4,5,6,7,8]]
t.model.predict([inp1, inp2]).shape