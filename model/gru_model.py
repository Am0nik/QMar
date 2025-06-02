#gru_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, TimeDistributed

def build_model(vocab_size, embedding_dim=128, gru_units=256):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    
    model.add(GRU(gru_units, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(GRU(gru_units, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    
    return model
