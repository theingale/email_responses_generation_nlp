# Necessary Imports
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from utils.preprocessors import input_vocab_len
from utils.preprocessors import output_vocab_len
from utils.constants import IP_SENT_MAXLENGTH
from utils.constants import OP_SENT_MAXLENGTH


# Load embedding matrix
enc_embedding_matrix = np.load('embeddings\encoder_emb_matrix.npy')
dec_embedding_matrix = np.load('embeddings\decoder_emb_matrix.npy')

# Define Model

# Define encoder
class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,
    encoder_final_state_h,encoder_final_state_c
    
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        
        super(Encoder, self).__init__()
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        #Initialize Embedding layer
        self.embedding_layer = Embedding(input_dim = self.inp_vocab_size,
                                         output_dim = self.embedding_size,
                                         input_length = self.input_length,
                                         embeddings_initializer=tf.initializers.Constant(enc_embedding_matrix),
                                         trainable=False)
        
        #Intialize Encoder LSTM layer
        self.lstm_layer = LSTM(self.lstm_size, return_state=True, return_sequences=True)

        self.lstm_output  = 0
        self.lstm_state_h = 0
        self.lstm_state_c = 0

    def call(self,input_sequence, states=None):
        '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the 
          embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
        '''
        input_emb = self.embedding_layer(input_sequence)
        self.lstm_output, self.lstm_state_h, self.lstm_state_c = self.lstm_layer(input_emb, states)
        return self.lstm_output, self.lstm_state_h, self.lstm_state_c
      

    def initialize_states(self, batch_size):
        '''
        Given a batch size it will return intial hidden state and intial cell state.
        If batch size is 32- Hidden state is zeros of size [32,lstm_units],
        cell state zeros is of size [32,lstm_units]
        '''
        initial_h = tf.zeros(shape = (batch_size, self.lstm_size))
        initial_c = tf.zeros(shape = (batch_size, self.lstm_size))
        states = [initial_h, initial_c]
        return states

class Decoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,out_vocab_size,embedding_size,lstm_size,input_length):
        
        super(Decoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        #Initialize Embedding layer
        self.decoder_embedding = Embedding(input_dim = self.out_vocab_size,
                                           output_dim = self.embedding_size,
                                           input_length = self.input_length,
                                           embeddings_initializer=tf.initializers.Constant(dec_embedding_matrix),
                                           trainable=False)
        #Intialize Decoder LSTM layer
        self.decoder_lstm = LSTM(self.lstm_size, return_sequences=True, return_state=True)
        #self.initial_states = 


    def call(self,input_sequence,initial_states=None):
        '''
        This function takes a sequence input and the initial states of the encoder.
        Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to decoder_lstm
        
        returns -- decoder_output,decoder_final_state_h,decoder_final_state_c
        '''

        decoder_emb = self.decoder_embedding(input_sequence)
        self.lstm_output, self.lstm_state_h, self.lstm_state_c = self.decoder_lstm(decoder_emb, initial_states)
        return self.lstm_output, self.lstm_state_h, self.lstm_state_c  

# Define model class
class Encoder_decoder(tf.keras.Model):
    
    def __init__(self,*params):
        
        super(Encoder_decoder, self).__init__()
        #Create encoder object
        self.encoder = Encoder(input_vocab_len, embedding_size=300, lstm_size=256,
                          input_length = IP_SENT_MAXLENGTH)
        #Create decoder object
        self.decoder = Decoder(output_vocab_len, embedding_size=300, lstm_size=256,
                          input_length = OP_SENT_MAXLENGTH)
        #Intialize Dense layer(out_vocab_size) with activation='softmax'
        self.dense = Dense(output_vocab_len, activation='softmax')

    
    def call(self, data):
        '''
        A. Pass the input sequence to Encoder layer -- Return encoder_output,encoder_final_state_h,encoder_final_state_c
        B. Pass the target sequence to Decoder layer with intial states as encoder_final_state_h,encoder_final_state_C
        C. Pass the decoder_outputs into Dense layer 
        
        Return decoder_outputs
        '''
        input, output = data[0], data[1]
        enc_op, enc_h, enc_c = self.encoder(input)
        dec_op, _, _ = self.decoder(output, [enc_h, enc_c]) 
        dense_op = self.dense(dec_op)
        return dense_op


# Cutom loss function to avoid computation for paddded values
# Ref: https://www.tensorflow.org/tutorials/text/image_captioning#model

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

def loss_function(real, pred):
    """ Custom loss function that will not consider the loss for padded zeros.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


@st.experimental_memo(show_spinner=False)
def build_model():
    """ Builds and prepared encoder-decoder model
    """

    # Build model object
    glove_model = Encoder_decoder()
    # Compile model
    glove_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=loss_function)
    # sample data
    enc_input = np.load('data\enc_input_sample.npy')
    dec_input = np.load('data\dec_input_sample.npy')
    dec_output = np.load('data\dec_output_sample.npy')
    # initialize model state
    glove_model.fit([enc_input, dec_input],dec_output, epochs=1, batch_size=128)
    print("Model Initialized Successfully")
    # load best model state
    glove_model.load_weights('model_weights\epoch-10_loss-1.091.h5')
    print("Restored weights Successfully")
    # return_model
    print("Model is ready...")
    return glove_model