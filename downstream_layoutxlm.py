from transformers import PreTrainedTokenizerBase, LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, AutoTokenizer
from transformers.file_utils import PaddingStrategy

import torch
from torch import nn

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union

import pdb

torch.cuda.set_device(1)

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

"""Load dataset, using huggingface datasets"""

from datasets import load_dataset

#pdb.set_trace()
dataset = load_dataset('R0bk/XFUN', 'xfun.zh')

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

train_batch_size = 4
test_batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=data_collator, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=data_collator)

batch = next(iter(train_dataloader))

"""And set our device"""

device = 'cuda'

"""And let's setup our new model"""

from transformers import LayoutLMv2ForTokenClassification

model = LayoutXLMForTokenClassification.from_pretrained('microsoft/layoutxlm-base',num_labels=7)

# import LayoutLMv2_RE
# model = LayoutLMv2_RE.LayoutLMv2ForRelationExtraction.from_pretrained('microsoft/layoutxlm-base')
model.to(device)

"""Now lets do some training, first setup the optimizer"""

from transformers import AdamW
from tqdm.notebook import tqdm

lr_bounds = (5e-10, 5e-5)
optimizer = AdamW(model.parameters(), lr=lr_bounds[0])

global_step = 0
num_train_epochs = 10
t_total = len(train_dataloader) * num_train_epochs # total number of training steps
lr_warmup_ratio = 0.15 # percentage of steps we want to warm up on
lr_warmup_steps = t_total*lr_warmup_ratio
print(f'total steps: {t_total}, total examples:, {t_total*train_batch_size}, warmup steps: {lr_warmup_steps}')

#put the model in training mode
model.train()
for epoch in range(num_train_epochs):  
    print(f'Epoch: {epoch}')
    avg_loss = 0.
    nb_examples = 0
    for batch in tqdm(train_dataloader):
        del batch['id']
        del batch['hd_image']
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

# """We can now do evaluation, first we can take the region scoring function from unilm"""

# def re_score(pred_relations, gt_relations, mode="strict"):
#     """Evaluate RE predictions
#     Args:
#         pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
#         gt_relations (list) :    list of list of ground truth relations
#             rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
#                     "tail": (start_idx (inclusive), end_idx (exclusive)),
#                     "head_type": ent_type,
#                     "tail_type": ent_type,
#                     "type": rel_type}
#         vocab (Vocab) :         dataset vocabulary
#         mode (str) :            in 'strict' or 'boundaries'"""

#     assert mode in ["strict", "boundaries"]

#     relation_types = [v for v in [0, 1] if not v == 0]
#     scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

#     # Count GT relations and Predicted relations
#     n_sents = len(gt_relations)
#     n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
#     n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

#     # Count TP, FP and FN per type
#     for pred_sent, gt_sent in zip(pred_relations, gt_relations):
#         for rel_type in relation_types:
#             # strict mode takes argument types into account
#             if mode == "strict":
#                 pred_rels = {
#                     (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
#                     for rel in pred_sent
#                     if rel["type"] == rel_type
#                 }
#                 gt_rels = {
#                     (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
#                     for rel in gt_sent
#                     if rel["type"] == rel_type
#                 }

#             # boundaries mode only takes argument spans into account
#             elif mode == "boundaries":
#                 pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
#                 gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

#             scores[rel_type]["tp"] += len(pred_rels & gt_rels)
#             scores[rel_type]["fp"] += len(pred_rels - gt_rels)
#             scores[rel_type]["fn"] += len(gt_rels - pred_rels)

#     # Compute per entity Precision / Recall / F1
#     for rel_type in scores.keys():
#         if scores[rel_type]["tp"]:
#             scores[rel_type]["p"] = scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
#             scores[rel_type]["r"] = scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
#         else:
#             scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

#         if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
#             scores[rel_type]["f1"] = (
#                 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
#             )
#         else:
#             scores[rel_type]["f1"] = 0

#     # Compute micro F1 Scores
#     tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
#     fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
#     fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

#     if tp:
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         f1 = 2 * precision * recall / (precision + recall)

#     else:
#         precision, recall, f1 = 0, 0, 0

#     scores["ALL"]["p"] = precision
#     scores["ALL"]["r"] = recall
#     scores["ALL"]["f1"] = f1
#     scores["ALL"]["tp"] = tp
#     scores["ALL"]["fp"] = fp
#     scores["ALL"]["fn"] = fn

#     # Compute Macro F1 Scores
#     scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
#     scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
#     scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

#     print(f"RE Evaluation in *** {mode.upper()} *** mode")

#     print(
#         "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(
#             n_sents, n_rels, n_found, tp
#         )
#     )
#     print(
#         "\tALL\t TP: {};\tFP: {};\tFN: {}".format(scores["ALL"]["tp"], scores["ALL"]["fp"], scores["ALL"]["fn"])
#     )
#     print("\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(precision, recall, f1))
#     print(
#         "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
#             scores["ALL"]["Macro_p"], scores["ALL"]["Macro_r"], scores["ALL"]["Macro_f1"]
#         )
#     )

#     for rel_type in relation_types:
#         print(
#             "\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
#                 rel_type,
#                 scores[rel_type]["tp"],
#                 scores[rel_type]["fp"],
#                 scores[rel_type]["fn"],
#                 scores[rel_type]["p"],
#                 scores[rel_type]["r"],
#                 scores[rel_type]["f1"],
#                 scores[rel_type]["tp"] + scores[rel_type]["fp"],
#             )
#         )

#     return scores

# def compute_metrics(p):
#     pred_relations, gt_relations = p
#     score = re_score(pred_relations, gt_relations, mode="boundaries")
#     return score

# """Now we can write a new function to iterate through the validation set"""

# # from datasets import load_metric
# from transformers.trainer_utils import EvalPrediction
# import numpy as np


# # put model in evaluation mode
# model.eval()
# label_names = ['labels', 'relations']
# metric_key_prefix = 'eval'
# re_labels = None
# pred_relations = None
# entities = None
# for batch in tqdm(test_dataloader, desc="Evaluating"):
#     with torch.no_grad():
#         del batch['id']
#         del batch['hd_image']
#         for k, v in batch.items():
#             if hasattr(v, "to") and hasattr(v, "device"):
#                 batch[k] = v.to(device)

#         # forward pass
#         outputs = model(**batch)
#         labels = tuple(batch.get(name) for name in label_names)


#         # Setup labels
#         re_labels = labels[1] if re_labels is None else re_labels + labels[1]
#         pred_relations = (
#             outputs.pred_relations if pred_relations is None else pred_relations + outputs.pred_relations
#         )
#         entities = outputs.entities if entities is None else entities + outputs.entities        
        
# gt_relations = []
# for b in range(len(re_labels)):
#     rel_sent = []
#     for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
#         rel = {}
#         rel["head_id"] = head
#         rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
#         rel["head_type"] = entities[b]["label"][rel["head_id"]]

#         rel["tail_id"] = tail
#         rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
#         rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

#         rel["type"] = 1

#         rel_sent.append(rel)

#     gt_relations.append(rel_sent)


# re_metrics = compute_metrics(EvalPrediction(predictions=pred_relations, label_ids=gt_relations))

# re_metrics = {
#     "precision": re_metrics["ALL"]["p"],
#     "recall": re_metrics["ALL"]["r"],
#     "f1": re_metrics["ALL"]["f1"],
# }
# re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()

# metrics = {}

# # # Prefix all keys with metric_key_prefix + '_'
# for key in list(re_metrics.keys()):
#     if not key.startswith(f"{metric_key_prefix}_"):
#         metrics[f"{metric_key_prefix}_{key}"] = re_metrics.pop(key)
#     else:
#         metrics[f"{key}"] = re_metrics.pop(key)

# print(metrics)



# """## Inference example

# First to understand the relation extraction model we have to understand that this model requires as inputs already extracted entities, we are just interested in linking entities in this model, not detecting them.

# So for an inference example we can first take an item from the validation set and remove everything except for `(input_ids, attention_mask, bbox, image, entities)`

# If you want to do inference on your own dataset you will have to train your own entity detection model first. Also to note is that this model predicts links from entities with label `1` to entities with label `2` so you should make sure that you can format the data from your entity detection model accordingly.

# The final entities dict should be in the format:


# ```
# {
#     'start': `torch.IntTensor` of shape `(num_entites)`,
#         Each value in the list represents the id of the token (element of range(0, len(tokens)) where the
#         entity starts
#     'end': `torch.IntTensor` of shape `(num_entites)`,
#         Each value in the list represents the id of the token (element of range(0, len(tokens)) where the
#         entity ends
#     'label': `torch.IntTensor` of shape `(num_entites)`
#         Each value in the list represents the label (as an int) of the entity
# }
# ```

# Now that we know this, let's take an example and show it
# """

# import copy
# batch = next(iter(test_dataloader))

# inf_example = {k: v[0] for k,v in batch.items()}
# inf_example.keys()

# inf_keep_keys = ['input_ids', 'bbox', 'hd_image', 'image', 'attention_mask', 'entities']
# new_inf_example = {k: copy.deepcopy(inf_example[k]) for k in inf_keep_keys}
# new_inf_example.keys()

# new_inf_example['entities'] = [new_inf_example['entities']]#[{'start': [], 'end': [], 'label': []}]
# new_inf_example['relations'] = [{'start_index': [], 'end_index': [], 'head': [], 'tail': []}]

# """Lets do a prediction"""

# with torch.no_grad():
#     del new_inf_example['hd_image']
#     for k, v in new_inf_example.items():
#         if hasattr(v, "to") and hasattr(v, "device"):
#             new_inf_example[k] = v.unsqueeze(0).to(device)

#     # forward pass
#     outputs = model(**new_inf_example)

# """And lets look at the predicted relations"""

# outputs['pred_relations'][0]

# """Now lets visualise them"""

# pil_img = to_img(torch.tensor(inf_example['hd_image'], dtype=torch.uint8))
# draw = ImageDraw.Draw(pil_img)

# rels = outputs['pred_relations'][0]
# ents = inf_example['entities']
# print(rels)
# print(ents)
# rels_by_index = [set(), set()]

# for rel in rels:
#     rels_by_index[0].update({x for x in range(*rel['head'])})
#     rels_by_index[1].update({x for x in range(*rel['tail'])})

# for i, (token, bbox) in enumerate(zip(inf_example['input_ids'], inf_example['bbox'])):
#     if token != 1:
#         x0 = bbox[0]*scale_factor
#         y0 = bbox[1]*scale_factor
#         x1 = bbox[2]*scale_factor
#         y1 = bbox[3]*scale_factor
#         draw.rectangle(((x0, y0), (x1, y1)), outline='red')
#         if i in rels_by_index[0]:
#             draw.rectangle(((x0, y0), (x1, y1)), outline='blue', width=2)
#         if i in rels_by_index[1]:
#             draw.rectangle(((x0, y0), (x1, y1)), outline='green', width=2)
        
# pil_img

