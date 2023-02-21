import PIL
from transformers import PreTrainedTokenizerBase, LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, AutoTokenizer
from transformers.file_utils import PaddingStrategy

import torch
from torch import nn

import numpy as np

import eval_metrics

from dataclasses import dataclass

from typing import Dict, Tuple, Optional, Union

# from datasets import load_metric
from transformers.trainer_utils import EvalPrediction

import pdb
import json

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--gpu", type=int, help="GPU ID")
args = argParser.parse_args()

torch.cuda.set_device(args.gpu)

feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

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

error_dict = {}

def model_eval(model, test_dataloader):
    model.eval()
    label_names = ['labels', 'relations','bbox','id']
    metric_key_prefix = 'eval'
    re_labels = None
    pred_relations = None
    entities = None
    bbox_gt = []
    id = None
    for batch in test_dataloader:
        with torch.no_grad():
            labels = tuple(batch.get(name) for name in label_names)
            del batch['id']
            del batch['hd_image']
            for k, v in batch.items():
                if hasattr(v, "to") and hasattr(v, "device"):
                    batch[k] = v.to(device)
            #pdb.set_trace()
            # forward pass
            outputs = model(**batch)
            
            # Setup labels
            re_labels = labels[1] if re_labels is None else re_labels + labels[1]
            pred_relations = (
                outputs.pred_relations if pred_relations is None else pred_relations + outputs.pred_relations
            )
            
            entities = outputs.entities if entities is None else entities + outputs.entities 
            for b in range(len(labels[2])):       
                bbox_gt.append({labels[3][b]:labels[2][b]})
            id = labels[3] if id is None else id + labels[3]

    gt_relations = []
    gt_line = []
    pred_line = []
    for b in range(len(re_labels)):
        rel_sent = []
        line_sent = []
        pred_line_sent = []
        for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
            rel = {}
            rel["head_id"] = head
            rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
            rel["head_type"] = entities[b]["label"][rel["head_id"]]
            
            #head entity bbox start
            (x0,y0) = (int(bbox_gt[b][id[b]][entities[b]["start"][rel["head_id"]]+1][0]),int(bbox_gt[b][id[b]][entities[b]["start"][rel["head_id"]]+1][3]))
            #head entity bbox end
            (x1,y1) = (int(bbox_gt[b][id[b]][entities[b]["end"][rel["head_id"]]-1][2]),int(bbox_gt[b][id[b]][entities[b]["end"][rel["head_id"]]-1][3]))
            
            rel["tail_id"] = tail
            rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
            rel["tail_type"] = entities[b]["label"][rel["tail_id"]]
            
            #tail entity bbox start
            (x2,y2) = (int(bbox_gt[b][id[b]][entities[b]["start"][rel["tail_id"]]+1][0]),int(bbox_gt[b][id[b]][entities[b]["start"][rel["tail_id"]]+1][3]))
            #tail entity bbox end
            (x3,y3) = (int(bbox_gt[b][id[b]][entities[b]["end"][rel["tail_id"]]-1][2]),int(bbox_gt[b][id[b]][entities[b]["end"][rel["tail_id"]]-1][3]))
            
            rel["type"] = 1

            rel_sent.append(rel)
            line_sent.append([(x0,y0),(x1,y1),(x2,y2),(x3,y3)])

        gt_relations.append(rel_sent)
        gt_line.append(line_sent)

        for rel_no in range(len(pred_relations[b])):
            rel = pred_relations[b][rel_no]
            #head entity bbox start
            (x0,y0) = (int(bbox_gt[b][id[b]][entities[b]["start"][rel["head_id"]]+1][0]),int(bbox_gt[b][id[b]][entities[b]["start"][rel["head_id"]]+1][3]))
            #head entity bbox end
            (x1,y1) = (int(bbox_gt[b][id[b]][entities[b]["end"][rel["head_id"]]-1][2]),int(bbox_gt[b][id[b]][entities[b]["end"][rel["head_id"]]-1][3]))
            #tail entity bbox start
            (x2,y2) = (int(bbox_gt[b][id[b]][entities[b]["start"][rel["tail_id"]]+1][0]),int(bbox_gt[b][id[b]][entities[b]["start"][rel["tail_id"]]+1][3]))
            #tail entity bbox end
            (x3,y3) = (int(bbox_gt[b][id[b]][entities[b]["end"][rel["tail_id"]]-1][2]),int(bbox_gt[b][id[b]][entities[b]["end"][rel["tail_id"]]-1][3]))
            
            pred_line_sent.append([(x0,y0),(x1,y1),(x2,y2),(x3,y3)])
        pred_line.append(pred_line_sent)

    #pdb.set_trace()
    re_metrics = eval_metrics.compute_metrics(EvalPrediction(predictions=pred_relations, label_ids=gt_relations))

    re_metrics = {
        "precision": re_metrics["ALL"]["p"],
        "recall": re_metrics["ALL"]["r"],
        "f1": re_metrics["ALL"]["f1"],
    }
    re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()

    metrics = {}

    # # Prefix all keys with metric_key_prefix + '_'
    for key in list(re_metrics.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            metrics[f"{metric_key_prefix}_{key}"] = re_metrics.pop(key)
        else:
            metrics[f"{key}"] = re_metrics.pop(key)

    print(metrics)
    error_dict['pred'] = pred_line
    error_dict['actual'] = gt_line
    error_dict['id'] = id
    return metrics

from datasets import load_dataset

dataset = load_dataset('R0bk/XFUN', 'xfun.zh')

#pdb.set_trace()
"""Create our tokenizer"""

tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutxlm-base', pad_token='<pad>')

"""Let's setup the dataloaders"""

from torch.utils.data import DataLoader
data_collator = DataCollatorForKeyValueExtraction(
    tokenizer,
    pad_to_multiple_of=8,
    padding=True,
    max_length=512,
)

train_dataset = dataset['train']
test_dataset = dataset['validation']

train_batch_size = 2
test_batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=data_collator, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=data_collator)

"""And set our device"""

device = 'cuda'


"""And let's setup our new model"""

from transformers import LayoutLMv2ForRelationExtraction
model = LayoutLMv2ForRelationExtraction.from_pretrained('/home/pritika/layoutXLM/functional/checkpoints5/checkpt-180')
model.to(device)

model_eval(model,test_dataloader)

error_file = open("error_rc.json",'w')
json.dump(error_dict, error_file, indent = 6, ensure_ascii=False)

