import os
import re
import spacy
import jieba
from multiprocessing import Pool, cpu_count

from spacy.lang.ru import Russian
nlp_ru= Russian()
# Load Spacy models
nlp_es = spacy.load('es_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

# Extracts raw text from each file 
codes_langs = ['zh','es', 'en', 'ru', 'ja'] #ISO code of languages: chinese, spanish, english, russian, japanese.
def extract_raw_texts(list_of_codes):
    raw_files_texts = {}
    path = "data/no_boilerplate/"
    all_file_names = [file for file in os.listdir(path) if file.endswith('.txt')] # enlists names of .txt files 
    for language in list_of_codes:
        files = [f for f in all_file_names if re.findall(language, f)] # separating files by language 
        for f in files:
            with open(path + f, "r", encoding="utf-8") as file:
                raw = file.read()
            raw_files_texts[f] = raw
            print(f"Extracted raw text from {f}")
    return raw_files_texts

# Tokenizer function
def tokenizer(text, model_lang):
    nlp = model_lang
    nlp.max_length = 9000000
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space and not token.is_punct and not token.is_digit]
    return tokens

def tokenize_text(item):
    filename, text = item
    print(f"Tokenizing {filename}...")
    if 'es' in filename:
        model_lang = nlp_es
        tokens = tokenizer(text, model_lang)
    elif 'en' in filename:
        model_lang = nlp_en
        tokens = tokenizer(text, model_lang)
    elif 'zh' in filename:
        punc = ["\n", ", "," ","，",": ",'。',"-",":", "(",")","'","\"","」","「",
                "？","﹔","　","：","！","、","《","》","』","『","[","]"]
        tok = jieba.lcut(text, cut_all=False)
        tokens = [t for t in tok if t not in punc]
    elif 'ru' in filename:
        model_lang = nlp_ru
        tokens = tokenizer(text, model_lang)
    else:
        tokens = []  # If language is not matched
    print(f"Finished tokenizing {filename}")
    return filename, tokens

def save_tokens(tokens_langs):
    output_dir = "data/tokenized"
    os.makedirs(output_dir, exist_ok=True)
    for filename, tokens in tokens_langs.items():
        save_path = os.path.join(output_dir, f"{filename}.tokens")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(tokens))
        print(f"Saved tokenized text for {filename}")

if __name__ == "__main__":
    print("Extracting raw texts...")
    raw_texts = extract_raw_texts(codes_langs)
    items = list(raw_texts.items())

    # Use multiprocessing to parallelize tokenization
    print("Starting tokenization...")
    with Pool(cpu_count()) as pool:
        results = pool.map(tokenize_text, items)

    # Convert results to dictionary
    tokens_langs = {filename: tokens for filename, tokens in results}

    # Save the tokenized texts
    print("Saving tokenized texts...")
    save_tokens(tokens_langs)
    print("Tokenization and saving complete.")
