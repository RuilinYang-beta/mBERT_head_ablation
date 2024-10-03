"""
Load a fine-tuned model, 
and evaluate it on the test set of XNLI **with regard to** each language.
The result is a vector of size (NUM_LANGS, ).

USAGE: 
```
# manually change which model to load, then
python eval_no_ablation.py
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

# --- manually change which model to evaluate ---
MODEL_PATH = "./tuned_model_en" 
# MODEL_PATH = "./tuned_model_all" 

model_abbr = "en" if MODEL_PATH == "./tuned_model_en" else "all"


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

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== evaluation =====
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

accuracy_matrix = np.zeros((NUM_LANGS,))
f1_matrix = np.zeros((NUM_LANGS,))

print("--- entering eval loop ---")
# for efficiency reason, 
# tokenize one language then evaluate models of different ablation on it
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
  
  print(f"No_Ablation-{lang}: acc {accuracy['accuracy']:.3f}; f1 {f1['f1']:.3f} ")
  accuracy_matrix[lang_idx] = accuracy['accuracy']
  f1_matrix[lang_idx] = f1['f1']
        
np.save(f'./results/baselines/{model_abbr}-NoAblation-accuracy.npy', accuracy_matrix)
np.save(f'./results/baselines/{model_abbr}-NoAblation-f1.npy', f1_matrix)