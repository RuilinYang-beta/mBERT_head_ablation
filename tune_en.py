"""
Load the wrangled and purified XNLI dataset, 
tokenize it for one language, 
and fine-tune mBERT on it.
"""
import time
import os
import numpy as np
import evaluate
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, EarlyStoppingCallback

from utils import generate_hyperparam_set

os.environ["TOKENIZERS_PARALLELISM"] = "false"

start_time = time.time()  
# ========= data =========
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
xnli = load_from_disk("./playground/xnli_purified/")

# ========= tokenizer and model =========
checkpoint = "google-bert/bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# export HF_ENDPOINT=https://hf-mirror.com

# ========= tokenization =========
# tokenize the dataset, in batch
# fine-tuning with English-only data 
def tokenize_batch_one_lang(batch, lang="en"):
    premise = [p[lang] for p in batch["premise"]]
    hypothesis = [h[lang] for h in batch["hypothesis"]]

    return tokenizer(premise, hypothesis, truncation=True)

xnli_tokenized = xnli.map(tokenize_batch_one_lang, batched=True) 

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

hyperparam_sets = [generate_hyperparam_set() for i in range(20)]

for idx, hyp in enumerate(hyperparam_sets): 
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

    print("==========================================")
    print(f"fine-tuning the {idx}-th model with the folloing hyperparams")
    print(hyp)
    

    training_args = TrainingArguments(
        output_dir=f"./trained_model-{idx}", 
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
