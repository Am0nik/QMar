from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(num_words=10000, oov_token="<OOV>", texts=None):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters='')
    special_tokens = ['<user>', '<assistant>']

    # Если передали корпус текстов, то обучаем на них вместе со спец.токенами
    if texts is not None:
        tokenizer.fit_on_texts(special_tokens + texts)
    else:
        tokenizer.fit_on_texts(special_tokens)

    return tokenizer

def tokenize_texts(tokenizer, texts, max_length=50):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post')
