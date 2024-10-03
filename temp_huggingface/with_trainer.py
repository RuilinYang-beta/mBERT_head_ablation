from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate

# ========= prepare =========
# data
raw_datasets = load_dataset("glue", "mrpc")

# tokenizer and model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# tokenize the dataset, in batch 
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) 

# batching for training 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ========= training =========
# more options: 
# * to evaluate during training, and 
# * to report metrics during evaluation

# provide a directory for saving the model
# check doc for more options
training_args = TrainingArguments("test-trainer")

trainer = Trainer(  
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train()

# ========= evaluation =========
predictions = trainer.predict(tokenized_datasets["validation"])
print(
    predictions.predictions,  # raw logits 
    predictions.label_ids,    # true labels
    predictions.metrics       # loss, time_consumed, custom metrics if passed `compute_metrics()` to Trainer
)

preds = np.argmax(predictions.predictions, axis=-1) # predicted label

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# ========= training with evaluation =========

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()