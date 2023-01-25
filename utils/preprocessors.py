# Necessary Imports
import joblib
import regex as re
from utils.constants import IP_SENT_MAXLENGTH
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Load Tokenizers
with open('tokenizers/input_tokenizer.pkl', 'rb') as handle:
    input_tokenizer = joblib.load(handle)

with open('tokenizers/output_tokenizer.pkl', 'rb') as handle:
    output_tokenizer = joblib.load(handle)

# Tokenizers constants
input_vocab_len = len(input_tokenizer.word_index) + 1
output_vocab_len = len(output_tokenizer.word_index) + 1

def clean_text(input_text):
    """converts text to lower case and removes unwanted punctuations, extra spaces,
       abbrevations etc. and returns clean text.

    Args:
        input_text (str): text input
    """
    # Defining funtion to preprocess data

    # lower case
    text = input_text.lower()

    # multiple occurences
    text = re.sub(r'\s+',' ', text)
    text = re.sub(r'\.{2,}','.', text)

    # Contractions
    # Ref: https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
    
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"c\'mon", "come on", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    

    # Removing unwanted words
    text = re.sub(r"hahaha", '', text)
    text = re.sub(r"wtf", '', text)
    text = re.sub(r"wth", '', text)
    text = re.sub(r"yep", 'yes', text)
    text = re.sub(r"sucks", 'is bad', text)
    text = re.sub(r"geez", '', text)
    text = re.sub(r"lol", '', text)

    # Don't remove ?
    text = text.replace("?", " ?")

    # Remove all other punctuations
    text = re.sub(r"[!\"#$%&'()*+,-\/:;<=>@[\]^_`{|}~]", '', text)

    # Remove emojis
    # Ref: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                      "]+", re.UNICODE)
    
    text = re.sub(emoj, '', text)

    return text

def preprocess_text(input_text):
    """Vectorizes and pads input text and returns padded sequence vector

    Args:
        input_text (str): clean input text
    """
    # tokenize text
    input = input_tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input, maxlen=IP_SENT_MAXLENGTH, padding='post')

    return input_sequence
