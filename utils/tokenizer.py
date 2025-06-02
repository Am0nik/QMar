# tokenizer.py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import matplotlib.pyplot as plt

def create_tokenizer(oov_token="<OOV>", texts=None):
    tokenizer = Tokenizer(oov_token=oov_token, filters='')
    special_tokens = ['<user>', '<assistant>', '<eos>']

    if texts is not None:
        tokenizer.fit_on_texts(special_tokens + texts)
    else:
        tokenizer.fit_on_texts(special_tokens)

    return tokenizer

def tokenize_texts(tokenizer, texts, max_length=50):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post')


