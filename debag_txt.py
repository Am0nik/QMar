import os

base_dir = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(base_dir, 'data', 'raw_text')

print("Содержимое папки raw_text:")
print(os.listdir(txt_path))
