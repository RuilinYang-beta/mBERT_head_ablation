from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

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

# ========= manually processes that Trainer automatically do =========
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names # ["attention_mask", "input_ids", "labels", "token_type_ids"]


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# ========= a quick check that everything is fine =========
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
# {'attention_mask': torch.Size([8, 65]),
#  'input_ids': torch.Size([8, 65]),
#  'labels': torch.Size([8]),
#  'token_type_ids': torch.Size([8, 65])}


outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)


optimizer = AdamW(model.parameters(), lr=5e-5)