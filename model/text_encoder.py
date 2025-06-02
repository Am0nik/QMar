#text_encoder.py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # <-- импорт для загрузки токенизатора
import json

class TextEncoder:
    def __init__(self, vocab_size=10000, oov_token="<OOV>"):
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        for token in ['<user>', '<assistant>', '<eos>']:
            if token not in self.tokenizer.word_index:
                index = len(self.tokenizer.word_index) + 1
                self.tokenizer.word_index[token] = index
                self.tokenizer.index_word[index] = token
    
        print(f"[TextEncoder] Фитинг токенизатора завершен. Размер словаря: {len(self.tokenizer.word_index)}")
        for token in ['<user>', '<assistant>', '<eos>']:
            if token in self.tokenizer.word_index:
                print(f"[TextEncoder] Токен '{token}' есть в словаре с индексом {self.tokenizer.word_index[token]}")
            else:
                print(f"[TextEncoder] Внимание: токен '{token}' отсутствует в словаре!")


    def encode(self, text, max_length=50):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')[0]
        #print(f"[TextEncoder] Кодирование текста: {text}")
        #print(f"[TextEncoder] Закодированная последовательность: {padded}")
        return padded

    def decode(self, tokens):
        reverse_word_index = dict([(index, word) for word, index in self.tokenizer.word_index.items()])
        words = []
        for i in tokens:
            if i == 0:
                continue
            word = reverse_word_index.get(i, '?')
            if word == '<eos>':
                break
            words.append(word)
        decoded = ' '.join(words)
        print(f"[TextEncoder] Декодирование токенов: {tokens}")
        print(f"[TextEncoder] Декодированный текст: {decoded}")
        return decoded


    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.tokenizer.to_json())
        print(f"[TextEncoder] Токенизатор сохранён в {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)  # <-- использование правильной функции
        print(f"[TextEncoder] Токенизатор загружен из {path}")
