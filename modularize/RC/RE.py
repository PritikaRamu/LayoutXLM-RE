import copy
import torch
from torch import nn
from RE_output import RegionExtractionOutput
from transformers.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2PreTrainedModel, LayoutLMv2Model
import pdb

class BiaffineAttention(nn.Module):
    """Implements a biaffine attention operator for binary relation classification.
    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.
    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.
    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.
    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class RegionExtractionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = nn.CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue

            #########################FIX#################################
            try:
                rel = {}
                rel["head_id"] = relations["head"][i]
                rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
                rel["head_type"] = entities["label"][rel["head_id"]]

                rel["tail_id"] = relations["tail"][i]
                rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
                rel["tail_type"] = entities["label"][rel["tail_id"]]
                rel["type"] = 1
                pred_relations.append(rel)
            except:
                continue
            #####################################################################
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations

class LayoutLMv2ForRelationExtraction(LayoutLMv2PreTrainedModel):
    def __init__(self, config, _configuration_file=None):
        super().__init__(config)
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = RegionExtractionDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        input_ids,
        bbox,
        labels=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        entities=None,
        relations=None,
    ):
        r"""
        entities (list of dicts of shape `(batch_size,)` where each dict contains:
            {
                'start': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in the list represents the id of the token (element of range(0, len(tokens)) where the
                    entity starts
                'end': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in the list represents the id of the token (element of range(0, len(tokens)) where the
                    entity ends
                'label': `torch.IntTensor` of shape `(num_entites)`
                    Each value in the list represents the label (as an int) of the entity
            }
        relations (list of dicts of shape `(batch_size,)` where each dict contains:
            {
                'head': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in the list represents the key of a different relation. A value can be used to map to
                    the entity list as it tells you what index to inspect in any of the lists inside the entities dict
                    (reps the id of the entity `(element of range(0, len(entities)`)
                'tail': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in the list represents the value of a different relation. A value can be used to map to
                    the entity list as it tells you what index to inspect in any of the lists inside the entities dict
                    (reps the id of the entity `(element of range(0, len(entities)`)
                'start_index': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in this list represents the start index (element of range(0, len(tokens)) for the
                    combined head and tail entities e.g. `min(entities['start']['head'], entities['start']['tail'])`
                'end_index': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in this list represents the end index (element of range(0, len(tokens)) for the
                    combined head and tail entities e.g. `min(entities['end']['head'], entities['end']['tail'])`
            }
        Returns:
        Examples:
        ```python
        >>> from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
        >>> from PIL import Image
        >>> processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        >>> model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")
        >>> words = ["hello", "world"]
        >>> boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
        >>> entities = *****
        >>> relations = *****
        ```"""
        #pdb.set_trace()
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        #pdb.set_trace()
        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return RegionExtractionOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )