import pandas as pd
import re
from sacrebleu import corpus_bleu
from collections import defaultdict
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
import numpy as np


#%% Cleaning and tok
# Clean text (lowercase, remove extra characters and spaces)
def clean_text(text):
    # Check if the value is None or not a string
    if text is None or not isinstance(text, str):
        return ""  # Return empty string for None values
    
    text = text.lower() # Lowercase for normalization
    # Remove unwanted characters, keep letters, numbers, spaces, and common punctuation
    text = re.sub(r"[^a-zäöüßA-ZÄÖÜẞ0-9\s.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Simple tokenizer: remove punctuation, split on whitespace
def ez_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    tokens = text.split()
    return tokens

# Tokenize sentences function using ez_tokenizer
def tokenize_sents(sents, nlp_model):
    tokenized = []
    for sent in sents:
        tokens = nlp_model(sent)
        tokenized.append(tokens)
    return tokenized
    
# Normalize text by lowercasing and stripping extra spaces
def normalize(sent):
    return sent.lower().strip()

# Reconstruct detokenized string, fixing common spacing/punctuation issues
def detokenize(text):
    # Remove any unecessary spaces before/after punctuation
    text = re.sub(r'\s+([?.!,:;»])', r'\1', text)
    text = re.sub(r'([«„])\s+', r'\1', text)

    # Fix common English contractions
    text = re.sub(r"\b(can|do|does|did|would|should|could|is|are|was|were|has|have|had|must|might|n't|won|n't|shan)\s+not\b", r"\1n't", text)
    text = re.sub(r"\bI\s+'m\b", "I'm", text)
    text = re.sub(r"\b([A-Za-z]+)\s+'(ll|re|ve|d|s|t)\b", r"\1'\2", text)

    # Remove double spaces if any
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


#%% Models
#Basic Unigram Model
def build_unigram_model(src_tokens, tgt_tokens):
    model = defaultdict(lambda: defaultdict(int))
    for src_tok, tgt_tok in zip(src_tokens, tgt_tokens):
        # Create bigrams for each sentence
        src_unigrams = list(ngrams(src_tok, 1))
        tgt_unigrams = list(ngrams(tgt_tok, 1))
        
        # Map each source bigram to each target bigram with a count
        # This simple approach increases counts for all possible pairs
        for s_unigram in src_unigrams:
            for t_unigram in tgt_unigrams:
                model[s_unigram][t_unigram] += 1
    return model

# Weighted Unigram Model (for custom model later)
def build_better_uni_model(src_tokens, tgt_tokens):
    model = defaultdict(lambda: defaultdict(float))

    src_freq = defaultdict(int)

    # Count source unigram frequencies
    for src_tok in src_tokens:
        src_unigrams = list(ngrams(src_tok, 1))
        for s_unigram in src_unigrams:
            src_freq[s_unigram] += 1

    for src_tok, tgt_tok in zip(src_tokens, tgt_tokens):
        src_unigrams = list(ngrams(src_tok, 1))
        tgt_unigrams = list(ngrams(tgt_tok, 1))
        
        n_src_grams = len(src_unigrams)
        n_tgt_grams = len(tgt_unigrams)
        for i in range(n_src_grams):
            for j in range(n_tgt_grams):
                s_unigram = src_unigrams[i]
                t_unigram = tgt_unigrams[j]
                diff = abs(i-j)

                weight = (-0.5)**diff + 1e-3
                model[s_unigram][t_unigram] += weight / (src_freq[s_unigram])

    # Normalize probabilities
    for s_unigram in model:
        total = sum(model[s_unigram].values())
        for t_unigram in model[s_unigram]:
            model[s_unigram][t_unigram] /= total

    return model


# Basic Bigram model 
def build_bigram_model(src_tokens, tgt_tokens):
    model = defaultdict(lambda: defaultdict(int))
    for src_tok, tgt_tok in zip(src_tokens, tgt_tokens):
        src_bigrams = list(ngrams(src_tok, 2))
        tgt_bigrams = list(ngrams(tgt_tok, 2))
        
        for s_bigram in src_bigrams:
            for t_bigram in tgt_bigrams:
                model[s_bigram][t_bigram] += 1
    return model

# Smoothed/Weighted Bigram Model with Gaussian-style alignment preference
def build_better_bi_model(src_tokens, tgt_tokens):
    model = defaultdict(lambda: defaultdict(float))

    src_freq = defaultdict(int)

    # Count source Bigram frequencies
    for src_tok in src_tokens:
        src_bigrams = list(ngrams(src_tok, 2))
        for s_bigram in src_bigrams:
            src_freq[s_bigram] += 1

    for src_tok, tgt_tok in zip(src_tokens, tgt_tokens):
        src_bigrams = list(ngrams(src_tok, 2))
        tgt_bigrams = list(ngrams(tgt_tok, 2))
        
        n_src_grams = len(src_bigrams)
        n_tgt_grams = len(tgt_bigrams)
        for i in range(n_src_grams):
            for j in range(n_tgt_grams):
                s_bigram = src_bigrams[i]
                t_bigram = tgt_bigrams[j]
                diff = abs(i-j)
                sigma = 0.5
                # "Gaussian" weighting -- value matching word order, but drop off to a low level for others, not as steep as -0.5**diff
                weight = np.exp(- (diff ** 2) / (2 * sigma ** 2))
                model[s_bigram][t_bigram] += weight / (src_freq[s_bigram])

    # Apply smoothing
    for s_bigram in model:
        total = sum(model[s_bigram].values()) + len(model[s_bigram])
        for t_bigram in model[s_bigram]:
            model[s_bigram][t_bigram] = (model[s_bigram][t_bigram] + 1) / total

    return model


# Translation using Unigram Model
def translate_unigram(sentence, model, tok):
    tokens = tokenize_sents([sentence], tok)[0]
    translated = []
    
    for i in range(len(tokens)):
        unigram = (tokens[i],)
        if unigram in model:
            best_target = max(model[unigram].items(), key=lambda x: x[1])[0]
            translated.append(best_target[0])
        else:
            translated.append(tokens[i])
    
    # Handle last token
    if len(tokens) == 1:
        translated = tokens
    
    return " ".join(translated)

# Translation using Bigram Model with Unigram fallback
def translate_bigram(sentence, model, unigram_model, tok):
    tokens = tokenize_sents([sentence], tok)[0]
    translated = []
    
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        if bigram in model:
            best_target = max(model[bigram].items(), key=lambda x: x[1])[0]
            if i == 0:
                translated.extend(best_target)
            else:
                translated.append(best_target[1])
        else:
            # Fallback to unigrams
            unigram = (tokens[i],)
            if unigram in unigram_model:
                best_target = max(unigram_model[unigram].items(), key=lambda x: x[1])[0]
                translated.append(best_target[0])
            else:
                translated.append(tokens[i])
    
    # Handle last token
    if len(tokens) == 1:
        translated = tokens
    
    return " ".join(translated)


#%% Evals

# Calculate word-level accuracy (exact match)
def word_accuracy(predictions, references):
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())  # Use set for exact word matching
        ref_tokens = set(ref.split())
        correct += len(pred_tokens.intersection(ref_tokens))
        total += len(ref_tokens)
    return correct / total if total > 0 else 0

# Calculate word-level precision and recall
def word_precision(predictions, references):
    pred_words = []
    ref_words = []
    for pred, ref in zip(predictions, references):
        pred_words.extend(pred.split())
        ref_words.extend(ref.split())
    pred_set = set(pred_words)
    ref_set = set(ref_words)
    
    true_positives = len(pred_set.intersection(ref_set))
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(ref_set) if len(ref_set) > 0 else 0
    return precision, recall

def word_f1_score(predictions, references):
    precision, recall = word_precision(predictions, references)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#%% main
nrows = 150000

tokenized_fname = f"tokenized_data_{nrows}.csv"

# Load Dataset
print(f"nrows: {nrows}")
df = pd.read_csv("./Final_Project/wmt14_translate_de-en_train.csv", delimiter=",", encoding="utf-8", nrows=nrows, engine="python", on_bad_lines="skip")

# Drop rows with None values
df= df.dropna(subset=['de', 'en'])
print(f'df length: {len(df)}')

# Apply cleaning
df["en"] = df["en"].apply(clean_text)
df["de"] = df["de"].apply(clean_text)

# Filter long or mismatched pairs
df = df[df['en'].str.split().str.len() < 50]
df = df[df['de'].str.split().str.len() < 50]

print(f'(cleaned) df length: {len(df)}')

# German and English section to lists
english_sents = df["en"].tolist()
german_sents = df["de"].tolist()


# Test set prep
eng_train, eng_test, ger_train, ger_test = train_test_split(
    english_sents, german_sents, test_size=0.2, random_state=42
)

# Tokenization
tok_en = ez_tokenizer
tok_de = ez_tokenizer
english_train_tokens = tokenize_sents(eng_train, tok_en)
german_train_tokens = tokenize_sents(ger_train, tok_de)

# Build Models
unigram_model = build_unigram_model(english_train_tokens, german_train_tokens)
bigram_model = build_bigram_model(english_train_tokens, german_train_tokens)
custom_unigram_model = build_better_uni_model(english_train_tokens, german_train_tokens)
custom_bigram_model = build_better_bi_model(english_train_tokens, german_train_tokens)

# Test input
eng_inputs = [
         "Hello",
         "How are you?",
         "I eat lots of cheese every day.",
         "The government has lots of power to make decisions.",
         "The process allows the use of both diplomatic and monetary policies.",
         "The Man",
         "The Woman"]
for eng_input in eng_inputs:
    print("\nEnglish version: ", eng_input)
    print("Unigram Translation:", translate_unigram(eng_input.lower(), unigram_model, tok_en))
    print("Bigram Translation:", translate_bigram(eng_input.lower(), bigram_model, unigram_model, tok_en))
    print("Custom Unigram Translation:", translate_unigram(eng_input.lower(), custom_unigram_model, tok_en))
    print("Custom Bigram Translation:", translate_bigram(eng_input.lower(), custom_bigram_model, custom_unigram_model, tok_en))
   

# Generate predictions
unigram_preds = [translate_unigram(s, unigram_model, tok_en) for s in eng_test]
bigram_preds = [translate_bigram(s, bigram_model, unigram_model, tok_en) for s in eng_test]
custom_unigram_preds = [translate_unigram(s, custom_unigram_model, tok_en) for s in eng_test]
custom_bigram_preds = [translate_bigram(s, custom_bigram_model, custom_unigram_model, tok_en) for s in eng_test]

# Normalize translations for comparison
unigram_preds_norm = [normalize(s) for s in unigram_preds]
bigram_preds_norm = [normalize(s) for s in bigram_preds]
custom_unigram_norm = [normalize(s) for s in custom_unigram_preds]
custom_bigram_norm = [normalize(s) for s in custom_bigram_preds]
ger_test_norm = [detokenize(normalize(s)) for s in ger_test]

# Prepare for BLEU
references = [ger_test_norm]
unigram_hyps = [detokenize(normalize(hyp)) for hyp in unigram_preds]
bigram_hyps = [detokenize(normalize(hyp)) for hyp in bigram_preds]
custom_unigram_hyps = [detokenize(normalize(hyp)) for hyp in custom_unigram_preds]
custom_bigram_hyps = [detokenize(normalize(hyp)) for hyp in custom_bigram_preds]

# Calculate BLEU scores
bleu_unigram = corpus_bleu(unigram_hyps, references)
bleu_bigram = corpus_bleu(bigram_hyps, references)
bleu_custom_unigram = corpus_bleu(custom_unigram_hyps, references)
bleu_custom_bigram = corpus_bleu(custom_bigram_hyps, references)

# Print BLEU
print(f"BLEU - Unigram Model: {bleu_unigram.score:.4f}")
print(f"BLEU - Bigram Model: {bleu_bigram.score:.4f}")
print(f"BLEU - Custom Unigram Model: {bleu_custom_unigram.score:.4f}")
print(f"BLEU - Custom Bigram Model: {bleu_custom_bigram.score:.4f}")

# Accuracy and F1
acc_unigram = word_accuracy(unigram_preds_norm, ger_test_norm)
acc_bigram = word_accuracy(bigram_preds_norm, ger_test_norm)
acc_custom_unigram = word_accuracy(custom_unigram_norm, ger_test_norm)
acc_custom_bigram = word_accuracy(custom_bigram_norm, ger_test_norm)

f1_unigram = word_f1_score(unigram_preds_norm, ger_test_norm)
f1_bigram = word_f1_score(bigram_preds_norm, ger_test_norm)
f1_custom_unigram = word_f1_score(custom_unigram_norm, ger_test_norm)
f1_custom_bigram = word_f1_score(custom_bigram_norm, ger_test_norm)

print("\nAccuracy and F1 Scores:")
print(f"Accuracy - Unigram Model: {acc_unigram:.4f}")
print(f"Accuracy - Bigram Model: {acc_bigram:.4f}")
print(f"Accuracy - Custom Unigram Model: {acc_custom_unigram:.4f}")
print(f"Accuracy - Custom Bigram Model: {acc_custom_bigram:.4f}")

print(f"F1 Score - Unigram Model: {f1_unigram:.4f}")
print(f"F1 Score - Bigram Model: {f1_bigram:.4f}")
print(f"F1 Score - Custom Unigram Model: {f1_custom_unigram:.4f}")
print(f"F1 Score - Custom Bigram Model: {f1_custom_bigram:.4f}")

print("\nNOTE:BLEU remains the primary metric for evaluating machine translation systems.")
