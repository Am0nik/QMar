#preprocessing_txt.py
from tensorflow.keras.preprocessing.text import Tokenizer

def parse_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('\n', '').strip()

    raw_dialogs = content.split("<user>")
    pairs = []
    
    for dialog in raw_dialogs:
        if "<assistant>" in dialog:
            user_part, assistant_part = dialog.split("<assistant>", 1)
            q = user_part.strip()
            a = assistant_part.replace("<eos>", "").strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def prepare_texts_for_tokenizer(filepath):
    pairs = parse_txt_file(filepath)
    texts = []
    for q, a in pairs:
        texts.append(f"<user> {q}")
        if not a.endswith('<eos>'):
            a += ' <eos>'
        texts.append(f"<assistant> {a}")
    # Добавляем специальные токены явно
    special_tokens = ['<user>', '<assistant>', '<eos>']
    return special_tokens + texts

def fit_tokenizer(texts, vocab_size=10000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


