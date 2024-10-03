"""
Load mBERT with randomly initialized classification heads, 
and evaluate it on the test set of XNLI **with regard to** each language.
The result is a matrix of size (NUM_SAMPLES, NUM_LANGS), 
where `NUM_SAMPLES` is the number of un fine-tuned mBERT.

USAGE: 
```
python eval_no_tune_no_ablation.py
```
"""
import os
import time 
import copy
import torch 
import numpy as np
import evaluate 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, DatasetDict

from utils import tokenize_batch_one_lang
from utils import eval_without_trainer_fast, ablate_head_in_place, ablate_layer_but_one_head_in_place

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_SAMPLES = 10  # sample 10 untuned mBERT 

checkpoint = "google-bert/bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# export HF_ENDPOINT=https://hf-mirror.com


# ---- constants ----
LANG_LIST = ['ar','bg','de','el','en',
             'es','fr','hi','ru','sw',
             'th','tr','ur','vi','zh']

NUM_LAYERS = 12
NUM_HEADS = 12
NUM_LANGS = 15

# features: ["premise", "hypothesis", "label"]
# where "premise" and "hypothesis" each map to 
# a dictionary of 15 language-translation pairs
xnli = load_from_disk("./playground/xnli_purified/")
xnli = xnli['test']  # only care about test set now

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== evaluation =====
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

accuracy_matrix = np.zeros((NUM_SAMPLES, NUM_LANGS))
f1_matrix = np.zeros((NUM_SAMPLES, NUM_LANGS))

print("--- entering eval loop ---")
for sample_idx in range(NUM_SAMPLES):
  # a model with a random head
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
  
  for lang_idx, lang in enumerate(LANG_LIST):
    # tokenize dataset of a language 
    eval_set = xnli.map(lambda batch: tokenize_batch_one_lang(tokenizer, batch, lang=lang), 
                        batched=True) 
    eval_set = eval_set.remove_columns(["premise", "hypothesis"])
    eval_set = eval_set.rename_column("label", "labels")
    eval_set.set_format("torch")
                        
    start_time = time.time()
    predictions, references = eval_without_trainer_fast(model, 
                                                        eval_set, 
                                                        data_collator)

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")

    end_time = time.time() 
    execution_time = end_time - start_time  # Calculate execution time
    print(f"Time of one evaluation: {execution_time:.4f} seconds")
    
    print(f"NoTune-NoAblation-{lang}: acc {accuracy['accuracy']:.3f}; f1 {f1['f1']:.3f} ")
    accuracy_matrix[sample_idx, lang_idx] = accuracy['accuracy']
    f1_matrix[sample_idx, lang_idx] = f1['f1']
        
np.save(f'./results/baselines/NoTune-NoAblation-accuracy.npy', accuracy_matrix)
np.save(f'./results/baselines/NoTune-NoAblation-f1.npy', f1_matrix)