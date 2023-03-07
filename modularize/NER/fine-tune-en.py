import numpy as np # linear algebra
import pandas as pd
import torch, torchvision
from datasets import load_dataset 
import pdb
from transformers import PreTrainedTokenizerBase, LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
from torch.utils.data import DataLoader
from transformers.file_utils import PaddingStrategy
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import pipeline
from PIL import Image
from transformers import LayoutXLMProcessor

import wandb

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--gpu", type=int, help="GPU ID")
argParser.add_argument("--run", type=int, help="run number")
argParser.add_argument("--lr", type=float, help="learning rate")
argParser.add_argument("--warm", type=float, help="warmup ratio")
argParser.add_argument("--batch", type=int, help="train batch size")
argParser.add_argument("--grad_acc", type=int, help="gradient accumulation steps")

args = argParser.parse_args()

torch.cuda.set_device(args.gpu)

wandb.init(
    # set the wandb project where this run will be logged
    project="Form-IE_NER_en",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "layoutXLM fine tune for NER",
    "dataset": 'funsd',
    "epochs": 200,
    "batch_size": args.batch,
    "warmup_ratio": args.warm,
    "output_dir": f"checkpoints{args.run}",
    }
)

dataset = load_dataset("nielsr/funsd")
labels = dataset['train'].features['ner_tags'].feature.names

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

from transformers import LayoutLMv2Processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

"""Create our tokenizer"""

#tokenizer = LayoutXLMProcessor.from_pretrained('microsoft/layoutxlm-base',apply_ocr=False)

features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
})

def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  words = examples['words']
  boxes = examples['bboxes']
  word_labels = examples['ner_tags']
  #pdb.set_trace()
  #encoded_inputs = tokenizer(images, text=words, boxes=boxes, word_labels=word_labels,
                        #     padding="max_length", truncation=True)
  test = processor(images, words, boxes=boxes, word_labels=word_labels,
                             padding="max_length", truncation=True)
  #print(encoded_inputs)
  return test

train_dataset = dataset['train'].map(preprocess_data, batched=True,features=features, remove_columns=dataset['train'].column_names)
test_dataset = dataset['test'].map(preprocess_data, batched=True,features=features, remove_columns=dataset['test'].column_names)
pdb.set_trace()
train_dataset.set_format(type="torch", device=f"cuda:{args.gpu}")
test_dataset.set_format(type="torch", device=f"cuda:{args.gpu}")

train_batch_size = args.batch
test_batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

from transformers import LayoutLMv2ForTokenClassification, AdamW

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutxlm-base',num_labels=len(labels))

device = 'cuda'
model.to(device)

"""Now lets do some training, first setup the optimizer"""

from transformers import AdamW

lr_bounds = (args.lr, 5e-5)
optimizer = AdamW(model.parameters(), lr=lr_bounds[0])

global_step = 0
num_train_epochs = 200
t_total = len(train_dataloader) * num_train_epochs # total number of training steps
lr_warmup_ratio = args.warm # percentage of steps we want to warm up on
lr_warmup_steps = t_total*lr_warmup_ratio

print(f'Hyperparameters:\ntrain_batch_size  = {train_batch_size}\ntest_batch_size = {test_batch_size}\nlr = {lr_bounds[0]}\nlr_warmup_ratio = {lr_warmup_ratio}')
print(f'total steps: {t_total}, total examples:, {t_total*train_batch_size}, warmup steps: {lr_warmup_steps}')

max_f1 = -1

#model eval code
from datasets import load_metric

metric = load_metric("seqeval")

def model_eval(model, test_dataloader):
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            #token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, labels=labels)
                            # token_type_ids=token_type_ids, labels=labels)
            
            # predictions
            predictions = outputs.logits.argmax(dim=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            metric.add_batch(predictions=true_predictions, references=true_labels)

    final_score = metric.compute()
    print(final_score)
    return final_score


#put the model in training mode
model.train()
for epoch in range(num_train_epochs):  
    print(f'Epoch: {epoch}')
    avg_loss = 0.
    nb_examples = 0
    for batch in train_dataloader:
        # del batch['id']
        # del batch['hd_image']
        # del batch['entities']
        # del batch['relations']
        #pdb.set_trace()
        for k, v in batch.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                batch[k] = v.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch) 
        loss = outputs.loss
        avg_loss += loss.item()
        nb_examples += batch['input_ids'].shape[0]
        
        # print loss every 10 steps
        if global_step % 10 == 0:
          print(f'Loss after {global_step} steps: {avg_loss/ nb_examples}')

        loss.backward()
        optimizer.step()
        global_step += 1

        # Do some learning rate warm up
        if global_step < lr_warmup_steps:
          for g in optimizer.param_groups:
              g['lr'] += (lr_bounds[1]-lr_bounds[0]) / lr_warmup_steps

    print(f'epoch done, avg_loss: {avg_loss/nb_examples}, nb_examples: {nb_examples}')
    wandb.log({"loss": avg_loss/nb_examples})

    metrics = model_eval(model, test_dataloader)
    wandb.log({'f1':metrics['overall_f1']})
    if metrics['overall_f1']>=max_f1:
        max_f1 = metrics['overall_f1']
        model.save_pretrained(f'checkpoints{args.run}/checkpt-{epoch}')

wandb.finish()