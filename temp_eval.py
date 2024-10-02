"""
Load a fine-tuned model, and evaluate it on the test set of XNLI.
"""
import copy
import torch 
import numpy as np
import evaluate 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, DatasetDict

from utils import eval_without_trainer, ablate_head_in_place

MODEL_PATH = "./en-2nd-iter/trained_model-9/checkpoint-432"

LANG_LIST = ['ar','bg','de','el','en',
             'es','fr','hi','ru','sw',
             'th','tr','ur','vi','zh']

NUM_LAYERS = 12
NUM_HEADS = 12
NUM_LANGS = 15

# ===== load data & model =====
xnli = load_from_disk("./playground/xnli_purified/")
xnli = xnli['test']  # only care about test set now

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ========= tokenization =========
def tokenize_batch(batch, lang="en"):
    premise = [p[lang] for p in batch["premise"]]
    hypothesis = [h[lang] for h in batch["hypothesis"]]

    return tokenizer(premise, hypothesis, truncation=True)

# ===== evaluation =====

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

accuracy_matrix = np.zeros((NUM_LAYERS, NUM_HEADS, NUM_LANGS+1))
f1_matrix = np.zeros((NUM_LAYERS, NUM_HEADS, NUM_LANGS+1))

print("--- entering eval loop ---")
for lang_idx, lang in enumerate(LANG_LIST):
    # tokenize dataset of a language 
    eval_set = xnli.map(lambda batch: tokenize_batch(batch, lang=lang), 
                        batched=True) 
    for num_layer in range(NUM_LAYERS): 
        for num_head in range(NUM_HEADS): 
            # make a copy of original model for each ablation
            model_copy = copy.deepcopy(model)
            ablate_head_in_place(model_copy, num_layer, num_head)
        
            predictions, references = eval_without_trainer(model_copy, eval_set)

            # Compute metrics
            accuracy = accuracy_metric.compute(predictions=predictions, references=references)
            f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")

            print(f"{num_layer}-{num_head}-{lang}: acc {accuracy['accuracy']:.3f}; f1 {f1['f1']:.3f} ")
            accuracy_matrix[num_layer, num_head, lang_idx] = accuracy['accuracy']
            f1_matrix[num_layer, num_head, lang_idx] = f1['f1']

np.save('./results/accuracy_matrix.npy', accuracy_matrix)
np.save('./results/f1_matrix.npy', f1_matrix)