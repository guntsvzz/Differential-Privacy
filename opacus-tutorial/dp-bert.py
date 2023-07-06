#!/usr/bin/env python
# coding: utf-8

'''
Differential Privacy (DP) with BERT model training

This is a modified, cleaned-up version from the Opacus notebook tutorial:
Building text classifier with Differential Privacy
https://github.com/pytorch/opacus/blob/main/tutorials/building_text_classifier.ipynb

Takes a pre-trained BERT-base model and fine-tunes on SNLI dataset
'''

import os
import argparse
import pickle
import numpy as np
import pandas as pd

import zipfile
import urllib.request
import warnings

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm

import transformers
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features

from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='bert model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_batch_size', type=int, default=8, help='max physical batch size')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--log_interval', type=int, default=5000, help='logging interval')
    parser.add_argument('--epsilon', type=float, default=7.5, help='epsilon DP')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='max grad norm')

    args = parser.parse_args()

    return args


# Dataset
# download the dataset (Stanford NLP mirror)
STANFORD_SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
DATA_DIR = "data"

warnings.simplefilter("ignore")

def download_and_extract(dataset_url, data_dir):
    print("Downloading and extracting ...")
    filename = "snli.zip"
    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(filename)
    print("Completed!")


snli_folder = os.path.join(DATA_DIR, 'snli_1.0')
# os.listdir(snli_folder)

if not os.path.exists(snli_folder):
    download_and_extract(STANFORD_SNLI_URL, DATA_DIR)
else:
    pass

train_path = os.path.join(snli_folder, "snli_1.0_train.txt")
dev_path = os.path.join(snli_folder, "snli_1.0_dev.txt")

df_train = pd.read_csv(train_path, sep='\t')
df_test = pd.read_csv(dev_path, sep='\t')

# df_train[['sentence1', 'sentence2', 'gold_label']][:5]

# get argparse
args = get_args()

# BERT Model
model_name = args.model_name

config = BertConfig.from_pretrained(
    model_name,
    num_labels=3,
)

tokenizer = BertTokenizer.from_pretrained(
    model_name,
    do_lower_case=False,
)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    config=config,
)


# Freeze layers
trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]

total_params = 0
trainable_params = 0

for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

for layer in trainable_layers:
    for p in layer.parameters():
        p.requires_grad = True
        trainable_params += p.numel()

print(f"Total parameters count: {total_params}") # ~108M
print(f"Trainable parameters count: {trainable_params}") # ~7M


# Preprocess the data
LABEL_LIST = ['contradiction', 'entailment', 'neutral']
MAX_SEQ_LENGTH = 128

def _create_examples(df, set_type):
    """ Convert raw dataframe to a list of InputExample. Filter malformed examples
    """
    examples = []
    for index, row in df.iterrows():
        if row['gold_label'] not in LABEL_LIST:
            continue
        if not isinstance(row['sentence1'], str) or not isinstance(row['sentence2'], str):
            continue

        guid = f"{index}-{set_type}"
        examples.append(
            InputExample(guid=guid, text_a=row['sentence1'], text_b=row['sentence2'], label=row['gold_label']))
    return examples


def _df_to_features(df, set_type):
    """ Pre-process text. This method will:
    1) tokenize inputs
    2) cut or pad each sequence to MAX_SEQ_LENGTH
    3) convert tokens into ids

    The output will contain:
    `input_ids` - padded token ids sequence
    `attention mask` - mask indicating padded tokens
    `token_type_ids` - mask indicating the split between premise and hypothesis
    `label` - label
    """
    examples = _create_examples(df, set_type)

    #backward compatibility with older transformers versions
    legacy_kwards = {}
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("2.9.0"):
        legacy_kwards = {
            "pad_on_left": False,
            "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            "pad_token_segment_id": 0,
        }

    return glue_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        label_list=LABEL_LIST,
        max_length=MAX_SEQ_LENGTH,
        output_mode="classification",
        **legacy_kwards,
    )


def _features_to_dataset(features):
    """ Convert features from `_df_to_features` into a single dataset
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )

    return dataset


train_features = _df_to_features(df_train, "train")
test_features = _df_to_features(df_test, "test")

train_dataset = _features_to_dataset(train_features)
test_dataset = _features_to_dataset(test_features)


# Choose batch size for DP
BATCH_SIZE = args.batch_size
MAX_PHYSICAL_BATCH_SIZE = args.max_batch_size
EPOCHS = args.epochs
LOGGING_INTERVAL = args.log_interval # once every how many steps we run evaluation cycle and report metrics

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

EPSILON = args.epsilon
DELTA = 1 / len(train_dataloader) # Parameter for privacy accounting. Probability of not achieving privacy guarantees
MAX_GRAD_NORM = args.max_grad_norm


# Training

# Move the model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to train mode (HuggingFace models load in eval mode)
model = model.train()

# ----------------------------------
# TODO: test to check/fix model
print(f'test if model is valid for DP: {ModuleValidator.is_valid(model)}')
# model = ModuleValidator.fix(model)
# ----------------------------------

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

def accuracy(preds, labels):
    return (preds == labels).mean()


# define evaluation cycle
def evaluate(model):
    model.eval()

    loss_arr = []
    accuracy_arr = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs['labels'].detach().cpu().numpy()

            loss_arr.append(loss.item())
            accuracy_arr.append(accuracy(preds, labels))

    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr)


# Attach privacy engine
privacy_engine = PrivacyEngine()

model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    target_delta=DELTA,
    target_epsilon=EPSILON,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

# Model training
for epoch in range(1, EPOCHS+1):
    losses = []

    with BatchMemoryManager(
        data_loader=train_dataloader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for step, batch in enumerate(tqdm(memory_safe_data_loader)):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':    batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}

            outputs = model(**inputs) # output = loss, logits, hidden_states, attentions

            loss = outputs[0]
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if step > 0 and step % LOGGING_INTERVAL == 0:
                train_loss = np.mean(losses)
                eps = privacy_engine.get_epsilon(DELTA)

                eval_loss, eval_accuracy = evaluate(model)

                print(
                  f"Epoch: {epoch} | "
                  f"Step: {step} | "
                  f"Train loss: {train_loss:.3f} | "
                  f"Eval loss: {eval_loss:.3f} | "
                  f"Eval accuracy: {eval_accuracy:.3f} | "
                  f"ɛ: {eps:.2f}"
                )