import csv
import re
import string

def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        leitor = csv.reader(csvfile)
        for linha in leitor:
            yield linha

def line_to_str(csv_line_list):
    return ",".join([word for word in csv_line_list])
    
def tokenize_text(text):
    # Remove pontuação usando o módulo string
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Converte o texto para letras minúsculas
    text = text.lower()
    
    # Divide o texto em tokens (palavras) usando espaços
    tokens = text.split()
    
    return tokens
