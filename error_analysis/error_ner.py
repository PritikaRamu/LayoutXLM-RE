import numpy as np # linear algebra
import pandas as pd
import torch, torchvision
from datasets import load_dataset 
import pdb
from transformers import PreTrainedTokenizerBase, LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, Optional, Union
from torch.utils.data import DataLoader
from transformers.file_utils import PaddingStrategy
from datasets import load_metric
import json
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from PIL import Image

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--gpu", type=int, help="GPU ID")
argParser.add_argument("--lang", type=str, help="dataset language")

args = argParser.parse_args()

torch.cuda.set_device(args.gpu)

error_dict = {}

if args.lang == 'zh':
    dataset = load_dataset('R0bk/XFUN', 'xfun.zh')
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

    test_dataset = dataset['validation']

    test_batch_size = 2

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=data_collator)

    from transformers import LayoutLMv2ForTokenClassification, AdamW

    model = LayoutLMv2ForTokenClassification.from_pretrained('/home/pritika/layoutXLM/modularize/NER/checkpoints1/checkpt-170',num_labels=len(labels))

    device = 'cuda'
    model.to(device)

    metric = load_metric("seqeval")

    def model_eval(model, test_dataloader):
        model.eval()
        for batch in test_dataloader:
            with torch.no_grad():
                #pdb.set_trace()
                
                input_ids = batch['input_ids'].to(device)
                bbox = batch['bbox'].to(device)
                image = batch['image'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # forward pass
                outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask)
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
                #
                for i in range(len(batch['id'])):
                    error_dict[batch['id'][i]] = {'pred':[],'actual':[],'bbox':[],'token':[]}
                    for j in range(len(true_labels[i])):
                        if (true_labels[i][j]!=true_predictions[i][j]):
                            error_dict[batch['id'][i]]['pred'].append(true_predictions[i][j])
                            error_dict[batch['id'][i]]['actual'].append(true_labels[i][j])
                            error_dict[batch['id'][i]]['bbox'].append(batch['bbox'][i][j].tolist())
                            error_dict[batch['id'][i]]['token'].append(tokenizer.decode(batch['input_ids'][i][j]))
                            print(tokenizer.decode(batch['input_ids'][i][j]))
                metric.add_batch(predictions=true_predictions, references=true_labels)

        final_score = metric.compute()
        print(final_score)
        return final_score

    model_eval(model,test_dataloader)
    #pdb.set_trace()
    error_file = open("error_ner.json",'w')
    json.dump(error_dict, error_file, indent = 6, ensure_ascii=False)

elif args.lang == 'en':
    dataset = load_dataset("nielsr/funsd")
    labels = dataset['train'].features['ner_tags'].feature.names
    dataset_file = open("funsd_val.json","w")
   
    bruh = {}
    bruh['id']= dataset['test']['id']
    bruh['words'] = dataset['test']['words']
    bruh['bboxes'] = dataset['test']['bboxes']
    bruh['ner_tags'] = dataset['test']['ner_tags']
    json.dump(bruh, dataset_file, indent = 6, ensure_ascii=False)

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

    test_dataset = dataset['test'].map(preprocess_data, batched=True,features=features, remove_columns=dataset['test'].column_names)

    test_dataset.set_format(type="torch", device=f"cuda:{args.gpu}")

    test_batch_size = 2

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

    from transformers import LayoutLMv2ForTokenClassification, AdamW

    model = LayoutLMv2ForTokenClassification.from_pretrained('/home/pritika/layoutXLM/modularize/NER/checkpoints4/checkpt-67',num_labels=len(labels))

    device = 'cuda'
    model.to(device)

    from datasets import load_metric

    metric = load_metric("seqeval")

    def model_eval(model, test_dataloader):
        model.eval()
        count = 0
        for batch in test_dataloader:
            with torch.no_grad():
                #pdb.set_trace()
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

                true_predictions_indexed = list()
                for prediction, label in zip(predictions, labels):
                    batch_wise = list()
                    index = 0
                    for (p, l) in zip(prediction, label):
                        if l!= -100:
                            batch_wise.append({index:id2label[p.item()]})
                        index += 1
                    true_predictions_indexed.append(batch_wise)

                true_labels = [
                    [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                #pdb.set_trace()

                for i in range(len(batch['input_ids'])):
                    error_dict[str(count)] = {'pred':[],'actual':[],'bbox':[],'token':[]}
                    for j in range(len(true_labels[i])):
                        if (true_labels[i][j]!=true_predictions[i][j]):
                            index = [l for l in true_predictions_indexed[i][j].keys()][0]
                            error_dict[str(count)]['pred'].append(true_predictions[i][j])
                            error_dict[str(count)]['actual'].append(true_labels[i][j])
                            error_dict[str(count)]['bbox'].append(batch['bbox'][i][index].tolist())
                            error_dict[str(count)]['token'].append({str(j):processor.decode(batch['input_ids'][i][index])})
                    count += 1
                metric.add_batch(predictions=true_predictions, references=true_labels)

        final_score = metric.compute()
        print(final_score)
        return final_score

    model_eval(model, test_dataloader)
    error_file = open("error_ner_funsd.json",'w')
    json.dump(error_dict, error_file, indent = 6, ensure_ascii=False)