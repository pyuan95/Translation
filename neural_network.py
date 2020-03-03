from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, ReLU, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.utils import plot_model
import re


class Translator:

    def __init__(self, language_seq_one, language_seq_two):
        self.PUNCTUATION = '([' + '!¡"#$%&()*,+-./:;<=>¿?@[\\]^`{|}~' + '])'
        self.start_seq = "START_SEQ__ "
        self.end_seq = " __END_SEQ"
        self.seq_one = language_seq_one
        self.seq_two = language_seq_two
        self.max_seq_one_len = 0
        self.max_seq_two_len = 0
        self.clean_sequences()
        self.set_max_seq_lengths()
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

    def set_max_seq_lengths(self):
        for s in self.seq_one:
            if len(s.split(" ")) > self.max_seq_one_len:
                self.max_seq_one_len = len(s.split(" "))

        for s in self.seq_two:
            if len(s.split(" ")) > self.max_seq_two_len:
                self.max_seq_two_len = len(s.split(" "))

    def space_punctuation(self, s):
        """ Pads a single space around all punctuation"""
        s = re.sub(self.PUNCTUATION, r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        return s

    def clean_sequences(self):
        for i, s in enumerate(self.seq_one):
            s = s.lower()
            s = self.start_seq + s + self.end_seq
            s = self.space_punctuation(s)
            self.seq_one[i] = s

        for i, s in enumerate(self.seq_two):
            s = s.lower()
            s = self.start_seq + s + self.end_seq
            s = self.space_punctuation(s)
            self.seq_two[i] = s

    def sequences_to_mat_generator(self):
        """
        Uses seq_two
        First axis of each training example
        Second axis is of each word in a training example
        Third axis is the word vector

        Pads sequences with the end sequence at the end
        """
        sequences = self.tokenizer_two.texts_to_sequences(self.seq_two)
        end_seq_index = self.tokenizer_two.texts_to_sequences([self.end_seq])[0][0]
        wordmat = np.zeros([self.max_seq_two_len, self.vocab_size_two + 1])
        wordmat_copy = wordmat.copy()

        i = 0
        while True:
            sequence = sequences[i]

            for j, word_index in enumerate(sequence):
                if j == 0:
                    continue
                wordmat_copy[j - 1][word_index] = 1
            while j < self.max_seq_two_len:
                wordmat_copy[j][end_seq_index] = 1
                j += 1
            yield wordmat_copy
            wordmat_copy = wordmat.copy()

            i += 1
            if i == len(sequences):
                i = 0

    def data_generator(self, batch_size=10):
        seq_one_encoded = self.tokenizer_one.texts_to_sequences(self.seq_one)
        seq_one_encoded = pad_sequences(seq_one_encoded, padding='post')
        seq_two_encoded = self.tokenizer_two.texts_to_sequences(self.seq_two)
        seq_two_encoded = pad_sequences(seq_two_encoded, padding='post')
        word_mat_gen = self.sequences_to_mat_generator()

        i = 0

        while True:
            x1_batch = []
            x2_batch = []
            y_batch = []
            for j in range(batch_size):
                x1_batch.append(seq_one_encoded[i])
                x2_batch.append(seq_two_encoded[i])
                y_batch.append(next(word_mat_gen))
                i += 1
                if i == len(seq_one_encoded):
                    i = 0

            yield [np.array(x1_batch), np.array(x2_batch)], np.array(y_batch)

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
        decoder_dense = Dense(self.vocab_size_two + 1, activation="softmax")
        decoder_dense_outputs = decoder_dense(decoder_LSTM_output)
        model = Model([encoder_inputs, decoder_inputs], decoder_dense_outputs)
        # plot_model(model, show_shapes=True)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')

        self.model = model
        return model

    def train_model(self):
        batch_size = 25
        gen = self.data_generator(batch_size=batch_size)

        [inp1, inp2], out = next(gen)
        inp1 = inp1[0]
        inp2 = inp2[0]
        out = out[0]
        # print(inp2)
        # print([np.argmax(x) for x in out])

        self.model.fit(x=self.data_generator(batch_size=batch_size), steps_per_epoch=len(self.seq_one) // batch_size,
                       epochs=30)
        self.model.save("model.h5")

    def predict_next_word(self, inp1, inp2):
        inp1 = np.array(self.tokenizer_one.texts_to_sequences([inp1]))
        inp2 = np.array(self.tokenizer_two.texts_to_sequences([inp2]))

        return self.tokenizer_two.sequences_to_texts([[np.argmax(self.model.predict([inp1, inp2])[0][-1])]])


def load_sequences_from_text_file(t):
    x = open(t, "r", encoding="utf8")
    seq_one = []
    seq_two = []
    index = 0
    for line in x:
        if index == 75000:
            break
        index += 1
        line = line.split("\t")
        seq_one.append(line[0])
        seq_two.append(line[1])
    return seq_one, seq_two

def predict_seq(t, seq):
    decoded_seq = t.start_seq.lower()
    new_word = t.predict_next_word(seq, decoded_seq)
    while new_word[0].lower() != t.end_seq.lower()[1:]:
        decoded_seq += " " + new_word[0]
        new_word = t.predict_next_word(seq, decoded_seq)
        print(decoded_seq)

    return decoded_seq

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(4)

seq_one, seq_two = load_sequences_from_text_file("spa.txt")
t = Translator(seq_one, seq_two)
t.initialize_model()
t.train_model()