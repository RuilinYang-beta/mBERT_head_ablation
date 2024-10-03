"""
Initialize a mBERT with random classification head, 
load the purified / purified_flattened XNLI dataset, 
and fine-tune the model on the loaded dataset, 
repeat this process <NUM_SETS> times to choose the best hyperparams.

USAGE: 
```
python tune.py <FINE-TUNE_MODE> -n <NUM_SETS> > log.txt
<FINE-TUNE_MODE>: en for fine-tuning on English-only data
                  all for fine-tuning on data of all languages
<NUM_SETS>: head for head ablation
            layer for layer ablation where only one head is left
Save the log to a file for later inspection.
```
"""
import time
import os
import numpy as np
import evaluate
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, EarlyStoppingCallback

from utils import prepare_tune_parser
from utils import generate_hyperparam_set
from utils import tokenize_batch_one_lang, tokenize_batch_all_lang

start_time = time.time()  
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === get the choice of modelfine-tuning mode and number of hyperparam sets ===
parser = prepare_tune_parser()
args = parser.parse_args()
print(args)

if args.model == "en": 
    DATA_PATH = "./data/xnli_purified"
    tokenize_fn = tokenize_batch_one_lang
    model_abbr = "en"

elif args.model == "all":
    DATA_PATH = "./data/xnli_purified_flattened" 
    tokenize_fn = tokenize_batch_all_lang
    model_abbr = "all"
else: 
    raise TypeError(f"Fine-tuning mode can only be 'en' or 'all'.")
            
NUM_SETS = args.num_sets

# ========= data =========
# --- "./data/xnli_purified/" ---
# `premise` and `hypothesis` are mapped to a dictionary of 15 language-translation pairs
# DatasetDict({
#     train: Dataset({
#         features: ['premise', 'hypothesis', 'label'],
#         num_rows: 5010
#     })
#     validation: Dataset({
#         features: ['premise', 'hypothesis', 'label'],
#         num_rows: 1245
#     })
#     test: Dataset({
#         features: ['premise', 'hypothesis', 'label'],
#         num_rows: 1245
#     })
# })
# --- "./data/xnli_purified_flattened/" ---
# `premise` and `hypothesis` each mapped to a translation of a certain language
# DatasetDict({
#     train: Dataset({
#         features: ['premise', 'hypothesis', 'label', 'lang'],
#         num_rows: 75150
#     })
#     validation: Dataset({
#         features: ['premise', 'hypothesis', 'label', 'lang'],
#         num_rows: 18675
#     })
#     test: Dataset({
#         features: ['premise', 'hypothesis', 'label', 'lang'],
#         num_rows: 18675
#     })
# })
xnli = load_from_disk(DATA_PATH)

# ========= tokenizer and model =========
checkpoint = "google-bert/bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# export HF_ENDPOINT=https://hf-mirror.com

# ========= tokenization =========

xnli_tokenized = xnli.map(lambda batch: tokenize_fn(tokenizer, batch), batched=True) 

# batching for training 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ========= training with evaluation =========
# --------- prep for evaluation ---------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# --------- training ---------
# provide a directory for saving the model

hyperparam_sets = [generate_hyperparam_set() for i in range(NUM_SETS)]

for idx, hyp in enumerate(hyperparam_sets): 
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

    print("==========================================")
    print(f"fine-tuning the {idx}-th model with the folloing hyperparams")
    print(hyp)
    

    training_args = TrainingArguments(
        output_dir=f"./trained_model-{model_abbr}-{idx}", 
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=0.05, 
        logging_strategy="steps",
        logging_steps=0.05,             # eval_steps defauts to be the same as logging_steps 
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,    
        save_total_limit=1,             # del older checkpoints

        **hyp
        )

    trainer = Trainer(  
        model,
        training_args,
        train_dataset=xnli_tokenized["train"],
        eval_dataset=xnli_tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  # Set patience to 2 epochs
    )

    print("------ before training ------")
    trainer.train()
    print("------ after training ------")

    # ------- evaluation -------
    predictions = trainer.predict(xnli_tokenized["test"])
    print(
        # predictions.predictions,  # raw logits 
        # predictions.label_ids,    # true labels
        predictions.metrics       # loss, time_consumed, custom metrics if passed `compute_metrics()` to Trainer
    )


# ========= timing =========
end_time = time.time()  # Record the end time

execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.4f} seconds")
