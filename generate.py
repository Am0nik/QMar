# generate.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Пути
MODEL_PATH = 'checkpoints/model.keras'
TOKENIZER_PATH = 'data/processed/vocab.json'
MAX_LEN = 50  # такой же, как при обучении

print("[generate.py] Загрузка модели...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[generate.py] Модель загружена.")

print("[generate.py] Загрузка токенизатора...")
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())
print("[generate.py] Токенизатор загружен.")

special_tokens = ['<user>', '<assistant>']
for token in special_tokens:
    if token not in tokenizer.word_index:
        print(f"Внимание: токен {token} отсутствует в словаре токенизатора.")
    else:
        print(f"Токен {token} присутствует в словаре, индекс: {tokenizer.word_index[token]}")

def sample_with_temperature(logits, temperature=1.0):
    # logits — это предсказания модели (логиты/вероятности)
    logits = logits.astype(np.float64)  # безопаснее в расчётах
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return np.random.choice(len(probs), p=probs)

def generate_reply(input_text, max_len=50, temperature=0.8):
    print(f"[generate_reply] Входной текст: {input_text}")
    if not input_text.startswith('<user>'):
        input_text = '<user> ' + input_text
        print(f"[generate_reply] Добавлен тег <user>: {input_text}")

    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')
    print(f"[generate_reply] Последовательность входа (индексы): {seq}")

    output_seq = []

    for i in range(max_len):
        preds = model.predict(seq, verbose=0)  # (1, vocab_size)
        print(f"Размер словаря токенизатора: {len(tokenizer.word_index)}")
        print(f"Максимальный индекс в словаре: {max(tokenizer.index_word.keys())}")
        print(f"Форма предсказаний модели: {preds.shape}")

        next_token_id = sample_with_temperature(preds[0, -1], temperature=temperature)

    
        print(f"[generate_reply] Шаг {i}, предсказанный токен: {next_token_id}")
    
        if next_token_id == 0:
            print("[generate_reply] Встречен токен padding (0), остановка генерации.")
            break
        
        word = tokenizer.index_word.get(next_token_id, '')
        print(f"[generate_reply] Токен соответствует слову: '{word}'")
    
        if word in ['<user>', '<assistant>']:
            print(f"[generate_reply] Встречен специальный токен '{word}', остановка генерации.")
            break
        
        output_seq.append(next_token_id)
    
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1] = next_token_id

    reply_text = ' '.join([tokenizer.index_word.get(i, '') for i in output_seq]).strip()
    print(f"[generate_reply] Сгенерированный текст: {reply_text}")

    if not reply_text.startswith('<assistant>'):
        reply_text = '<assistant> ' + reply_text

    return reply_text


if __name__ == "__main__":
    print("Введите 'выход' для завершения.")
    while True:
        user_input = input("Ты: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
        response = generate_reply(user_input, temperature=0.8)
        print("ИИ:", response.replace('<assistant>', '').strip())

