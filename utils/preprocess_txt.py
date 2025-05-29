
# preprocess_txt.py
from tensorflow.keras.preprocessing.text import Tokenizer

def parse_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        q, a = lines[i].strip(), lines[i + 1].strip()
        if q and a:
            pairs.append((q, a))
    return pairs

def prepare_texts_for_tokenizer(filepath):
    pairs = parse_txt_file(filepath)
    texts = []
    for q, a in pairs:
        texts.append(q)
        texts.append(a)
    special_tokens = ['<user>', '<assistant>']
    texts = special_tokens + texts
    return texts

def fit_tokenizer(texts, vocab_size=10000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer