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
    project="Form-IE_NER_de",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "layoutXLM fine tune for NER",
    "dataset": 'xfun.de',
    "epochs": 200,
    "batch_size": args.batch,
    "warmup_ratio": args.warm
    }
)

dataset = load_dataset('R0bk/XFUN', 'xfun.de')
labels = dataset['train'].features['labels'].feature.names

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}


feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)




"""Create our tokenizer"""

tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutxlm-base', pad_token='<pad>')

@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        has_image_input = "image" in features[0]
        has_bbox_input = "bbox" in features[0]
        if has_image_input:
            image = feature_extractor([torch.tensor(feature["image"]) for feature in features], return_tensors="pt")['pixel_values']
            # image = ImageList.from_tensors([torch.tensor(feature["image"]) for feature in features], 32)
            for feature in features:
                del feature["image"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        if has_image_input:
            batch["image"] = image
        return batch

data_collator = DataCollatorForKeyValueExtraction(
    tokenizer,
    pad_to_multiple_of=8,
    padding=True,
    max_length=512,
)

train_dataset = dataset['train']
test_dataset = dataset['validation']

train_batch_size = args.batch
test_batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=data_collator)

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
        del batch['id']
        del batch['hd_image']
        del batch['entities']
        del batch['relations']
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