import argparse
import math
import random 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_tune_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("model",
                      choices=['all', 'en'],
                      help="""
                      choose on which data to fine-tune mBERT:
                      all - data of all languages
                      en  - data of English language only 
                      """
                      ) 
  parser.add_argument("-n", "--num_sets", 
                      type=int, default=10,
                      help="number of sets of hyperparameters to generate"
                      )
  return parser

def prepare_eval_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("model",
                      choices=['all', 'en'],
                      help="choose which model to evaluate"
                      )  

  parser.add_argument("ablation", 
                      choices=["head", "layer"],
                      help="choose ablation mode"
                      )

  return parser


def tokenize_batch_one_lang(tokenizer, batch, lang="en"):
  """
  Tokenize premises and hypothesis of a chosen language.
  Apply to dataset with ['premise', 'hypothesis', 'label'] features,
  where 'premise' and 'hypothesis' each is a dictionary
  of 15 language-tranlation pairs.
  """
  premise = [p[lang] for p in batch["premise"]]
  hypothesis = [h[lang] for h in batch["hypothesis"]]

  return tokenizer(premise, hypothesis, truncation=True)

def tokenize_batch_all_lang(tokenizer, batch):
  """
  Tokenize premises and hypothesis of all languages.
  Apply to flattened dataset where each of 'premise' and 'hypothesis' 
  map to a value of translation of a certain language.
  """
  return tokenizer(batch["premise"], batch["hypothesis"], truncation=True)


def generate_hyperparam_set(): 
  """
  Generate a random hyperparameter set for hyperparameter tuning.
  These are the tune-able hyperparameters that we cherry pick from all possible ones.
  """
  # ------- fine-tune on en only: 1st iter -------
  # learning_rate = _sampling_on_log_scale(2e-5, 5e-5)
  # warmup_ratio = _sampling_on_linear_scale(0.1, 0.3)
  # weight_decay = _sampling_on_linear_scale(0.05, 0.3)

  # ------- fine-tune on en only: 2nd iter -------
  # learning_rate = _sampling_on_log_scale(2.3e-5, 3.5e-5)
  # warmup_ratio = _sampling_on_linear_scale(0.14, 0.2)
  # weight_decay = _sampling_on_linear_scale(0.05, 0.3)


  # ------- fine-tune on all lang: 1st iter -------
  # learning_rate = _sampling_on_linear_scale(1e-5, 4e-5)
  # warmup_ratio = _sampling_on_linear_scale(0.1, 0.4)
  # weight_decay = _sampling_on_linear_scale(0.1, 0.3)

  # ------- fine-tune on all lang: 2nd iter -------
  learning_rate = _sampling_on_linear_scale(1e-5, 2.5e-5)
  warmup_ratio = _sampling_on_linear_scale(0.15, 0.25)
  weight_decay = _sampling_on_linear_scale(0.18, 0.26)

  return {
    "learning_rate": learning_rate, 
    "warmup_ratio":  warmup_ratio, 
    "weight_decay":  weight_decay
    }

def _sampling_on_log_scale(lower_bound=0.0001, upper_bound=0.1):
  """
  Return a random number in range (lower_bound, upper_bound)
  To give smaller number more chances to be selected, 
  sampling on a log scale. 
  """
  lower = math.log10(lower_bound)
  upper = math.log10(upper_bound)

  r = random.uniform(lower, upper)
  return 10 ** r

import random

def _sampling_on_linear_scale(lower_bound=0.05, upper_bound=0.3):
    """
    Return a random number in the range (lower_bound, upper_bound) on a linear scale.
    """
    return random.uniform(lower_bound, upper_bound)


def eval_without_trainer(model, eval_set): 
  predictions = []
  references = []

  model.eval()
  with torch.no_grad():
    for example in eval_set:
        inputs = {k: torch.tensor(v).unsqueeze(0) 
                    for k, v in example.items() 
                        if k in ["input_ids", "token_type_ids", "attention_mask"]}
        
        outputs = model(**inputs)
        
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        
        predictions.append(predicted_label)
        references.append(example["label"]) 

  return  predictions, references

def eval_without_trainer_fast(model, eval_set, data_collator, batch_size=32): 
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  dataloader = DataLoader(eval_set, batch_size=batch_size, 
                          collate_fn=data_collator, shuffle=False)
  predictions = []
  references = []
    
  with torch.no_grad():
    for batch in tqdm(dataloader):  
        inputs = {k: v.clone().detach().to(device) 
                    for k, v in batch.items() 
                        if k in ["input_ids", "token_type_ids", "attention_mask"]}
        
        outputs = model(**inputs)

        predicted_labels = torch.argmax(outputs.logits, dim=1)
        
        predictions.extend(predicted_labels.cpu().numpy())
        references.extend(batch["labels"].cpu().numpy()) 

  return  predictions, references

# Access the attention layers
def ablate_head_in_place(model, layer_to_ablate, head_to_ablate):
  """
  Set attention head weights and biases in a specific layer to zero (ablating the head).
  """
  head_size = 64  # Size of each attention head (768 / 12 heads = 64)
  start_idx = head_to_ablate * head_size
  end_idx = (head_to_ablate + 1) * head_size

  with torch.no_grad():
    # Ablate the weights for query, key, value
    model.bert.encoder.layer[layer_to_ablate].attention.self.query.weight[:, start_idx:end_idx].zero_()
    model.bert.encoder.layer[layer_to_ablate].attention.self.key.weight[:, start_idx:end_idx].zero_()
    model.bert.encoder.layer[layer_to_ablate].attention.self.value.weight[:, start_idx:end_idx].zero_()

    # Ablate the biases for query, key, value
    model.bert.encoder.layer[layer_to_ablate].attention.self.query.bias[start_idx:end_idx].zero_()
    model.bert.encoder.layer[layer_to_ablate].attention.self.key.bias[start_idx:end_idx].zero_()
    model.bert.encoder.layer[layer_to_ablate].attention.self.value.bias[start_idx:end_idx].zero_()


def ablate_layer_but_one_head_in_place(model, layer_to_ablate, head_to_keep):
  """
  Set attention head weights and biases in a specific layer to zero (ablating the head).
  """
  num_heads = 12
  head_size = 64  # Size of each attention head (768 / 12 heads = 64)
  hidden_size = 768
  start_idx = head_to_keep * head_size
  end_idx = (head_to_keep + 1) * head_size
  
  mask = torch.zeros(hidden_size)
  mask[start_idx:end_idx] = 1

  with torch.no_grad():
    # Ablate the weights for query, key, value
    model.bert.encoder.layer[layer_to_ablate].attention.self.query.weight[:,:] *= mask
    model.bert.encoder.layer[layer_to_ablate].attention.self.key.weight[:,:] *= mask
    model.bert.encoder.layer[layer_to_ablate].attention.self.value.weight[:,:] *= mask

    # Ablate the biases for query, key, value
    model.bert.encoder.layer[layer_to_ablate].attention.self.query.bias[:] *= mask
    model.bert.encoder.layer[layer_to_ablate].attention.self.key.bias[:] *= mask
    model.bert.encoder.layer[layer_to_ablate].attention.self.value.bias[:] *= mask
