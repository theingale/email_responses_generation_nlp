# Necessary imports
import numpy as np
from utils.preprocessors import output_tokenizer



def predict(input_sequence, model, return_bigram_probas = False):

    '''
    A. Given input sentence the function returns predicted response
    '''
    pred_sent = []
    
    # get encoder states
    enc_op, state_h, state_c = model.layers[0](input_sequence)
    
    encoder_states = [state_h, state_c]

    target_word=np.zeros((1,1))
    target_word[0,0]=1 #<start> token
    
    # bigram word probabilities
    bigram_probas = []

    stop = False
    # k = 0
    
    while not stop:    
        dec_op, state_h, state_c = model.layers[1](target_word, encoder_states)
        encoder_states = [state_h, state_c]
        dense_op = model.layers[2](dec_op)
        word_id = np.argmax(dense_op, axis=-1)[0][0]
        #print(word_id)
        word_proba = np.max(dense_op, axis=-1)[0][0]
        bigram_probas.append(word_proba)
        target_word = word_id.reshape(1,1)
        word = output_tokenizer.index_word.get(word_id)
        if word == '<end>':
            stop = True
        else:
            pred_sent.append(str(word))    
            bigram_probas.append(word_proba)
    
    if return_bigram_probas:
        return ' '.join(pred_sent), bigram_probas
    else:
        return ' '.join(pred_sent)