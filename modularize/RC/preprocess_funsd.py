import json
import os
from datasets import DatasetDict, Dataset
import pdb
import cv2
import numpy as np

from transformers import LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained("/home/pritika/layoutXLM/layoutxlm-base")

train_anno_path = '/home/pritika/annotations'
train_image_path = '/home/pritika/images'




dataset = DatasetDict()

label2id = {'b-other':0,'i-other':0,'b-question':1,'b-answer':2,'b-header':3,'i-question':4,'i-answer':5,'i-header':6}


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def create_dataset(anno_path, image_path):
    train_dataset_dict = {'id':[],'input_ids':[],'bbox':[],'labels':[],'entities':[],'relations':[],'image':[]}
    anno_list = os.listdir(anno_path)
    for file in anno_list:
        with open(anno_path+'/'+file) as anno_file:
            anno = json.load(anno_file)
            img = cv2.imread(f'{image_path}/{file[:-5]}.png')
            h,w,c = img.shape
            img =  cv2.resize(img,(224,224))
            img = np.moveaxis(img,[0,1],[-2,-1]) 
            img = img.tolist()
            #to split files greater than 512
            
            split_count = 0
            train_dataset_dict['id'].append(file[:-5]+f'_{split_count}')
            train_dataset_dict['image'].append(img)
            #pre tokenized lists
            word_list = []
            label_list = []
            bbox_list = []
            temp_id_list = []

            relation_list = []
            entities_dict = {'start':[],'end':[],'label':[], 'link_id':[]}
            relations_dict = {'head':[], 'tail':[], 'start_index':[],'end_index':[]}
            
            for index in range(len(anno['form'])):

                #identify label of set of words
                label = anno['form'][index]['label']

                #id used to keep track of links (pre tokenized, refers to entity number)
                id = anno['form'][index]['id']

                # create list of relations for a doc
                if len(anno['form'][index]['linking'])!=0:
                    relation_list.append(anno['form'][index]['linking'])

                for i in range(len(anno['form'][index]['words'])):
                    word_list.append(anno['form'][index]['words'][i]['text'])
                    bbox = normalize_bbox(anno['form'][index]['words'][i]['box'],w,h)
                    bbox_list.append(bbox) 

                    if i==0:
                        label_list.append(label2id[f'b-{label}'])                        
                        temp_id_list.append(id)

                    else:
                        label_list.append(label2id[f'i-{label}'])
                        temp_id_list.append(id)

            embeddings = tokenizer(text=word_list, boxes=bbox_list, word_labels=label_list)

            temp_embed_for_id = tokenizer(text=word_list, boxes=bbox_list, word_labels=temp_id_list)

            #to keep track of number of input_ids
            ip_count = 0
            ip_list = []
            bb_list = []
            lab_list = []
            
            is_first_entity = True
            for index in range(len(embeddings['input_ids'])):
                if(ip_count>511):
                    break
                ip_list.append(embeddings['input_ids'][index])
                bb_list.append(embeddings['bbox'][index])
                lab_list.append(embeddings['labels'][index])
                if(embeddings['labels'][index]!=-100 and is_first_entity and embeddings['labels'][index] not in {0,3,6}):
                    entities_dict['start'].append(ip_count)
                    entities_dict['label'].append(embeddings['labels'][index])
                    entities_dict['link_id'].append(temp_embed_for_id['labels'][index])
                    is_first_entity = False
                elif(embeddings['labels'][index]!=-100 and is_first_entity!=True):
                    if(embeddings['labels'][index]-3!=entities_dict['label'][-1] and embeddings['labels'][index] not in {0,3,6}):
                        entities_dict['start'].append(ip_count)
                        entities_dict['label'].append(embeddings['labels'][index])
                        entities_dict['link_id'].append(temp_embed_for_id['labels'][index])
                        entities_dict['end'].append(ip_count)
                if(index==len(embeddings['input_ids'])-1):
                    entities_dict['end'].append(ip_count)
                ip_count += 1
            
            train_dataset_dict['input_ids'].append(ip_list)

            train_dataset_dict['labels'].append(lab_list)
            train_dataset_dict['bbox'].append(bb_list)
                
            train_dataset_dict['entities'].append(entities_dict)
            
            final_rel_list = []
            [final_rel_list.append(x[0]) for x in relation_list if x[0] not in final_rel_list]
            for rel in final_rel_list:
                try:
                    relations_dict['head'].append(entities_dict['link_id'].index(rel[0]))
                    relations_dict['tail'].append(entities_dict['link_id'].index(rel[1]))
                    relations_dict['start_index'].append(entities_dict['start'][entities_dict['link_id'].index(rel[0])])
                    relations_dict['end_index'].append(entities_dict['start'][entities_dict['link_id'].index(rel[1])])
                except:
                    continue
            
            train_dataset_dict['relations'].append(relations_dict)
    
    dataset = Dataset.from_dict(train_dataset_dict)
    return dataset

# data = create_dataset(train_anno_path,train_image_path)

# for i in range(len(data)):
#     if(len(data[i]['input_ids'])>512):
#         print('NOOOOO')