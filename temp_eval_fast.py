"""
Load a fine-tuned model, and evaluate it on the test set of XNLI.
"""
import os
import time 
import copy
import torch 
import numpy as np
import evaluate 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, DatasetDict

from utils import eval_without_trainer_fast, ablate_head_in_place, ablate_layer_but_one_head_in_place

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- choose what model to ablate ---
# MODEL_PATH = "./tuned_model_all/"
MODEL_PATH = "./tuned_model_en"
# -----------------------------------

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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
# for efficiency reason, 
# tokenize one language then evaluate models of different ablation on it
for lang_idx, lang in enumerate(LANG_LIST):
    # tokenize dataset of a language 
    eval_set = xnli.map(lambda batch: tokenize_batch(batch, lang=lang), 
                        batched=True) 
    eval_set = eval_set.remove_columns(["premise", "hypothesis"])
    eval_set = eval_set.rename_column("label", "labels")
    eval_set.set_format("torch")

    for num_layer in range(NUM_LAYERS): 
        for num_head in range(NUM_HEADS): 
            start_time = time.time()
            # make a copy of original model for each ablation
            model_copy = copy.deepcopy(model)

            # ----- two alternatives of ablation -----
            # ablate_head_in_place(model_copy, 
            #                      layer_to_ablate=num_layer, 
            #                      head_to_ablate=num_head)

            ablate_layer_but_one_head_in_place(model_copy, 
                                               layer_to_ablate=num_layer,
                                               head_to_keep=num_head)

            # -----------------------------------------
            
            predictions, references = eval_without_trainer_fast(model_copy, 
                                                                eval_set, 
                                                                data_collator)

            # Compute metrics
            accuracy = accuracy_metric.compute(predictions=predictions, references=references)
            f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")

            end_time = time.time() 
            execution_time = end_time - start_time  # Calculate execution time
            print(f"Time of one evaluation: {execution_time:.4f} seconds")
            

            print(f"{num_layer}-{num_head}-{lang}: acc {accuracy['accuracy']:.3f}; f1 {f1['f1']:.3f} ")
            accuracy_matrix[num_layer, num_head, lang_idx] = accuracy['accuracy']
            f1_matrix[num_layer, num_head, lang_idx] = f1['f1']
        
    # save per language
    np.save('./results/sample-one-lang.npy', accuracy_matrix)
    break
    # np.save('./results/en-layer-f1.npy', f1_matrix)