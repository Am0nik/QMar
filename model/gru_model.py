#gru_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, TimeDistributed, Bidirectional, BatchNormalization

def build_model(vocab_size, embedding_dim=128, gru_units=128):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))

    model.add(Bidirectional(GRU(gru_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Bidirectional(GRU(gru_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Bidirectional(GRU(gru_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
